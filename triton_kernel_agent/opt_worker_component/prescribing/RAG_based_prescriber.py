# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RAG-based prescriber for retrieving optimization patterns from the database.

This module provides embedding-based retrieval of optimization patterns from
a hierarchical knowledge base. The database is organized in 3 levels:
- L1: Bottleneck types (Compute, Memory, Under-utilization)
- L2: Optimization techniques (AIU load, tiling, attention overlap, ...)
- L3: Code examples (matmul, fused attention, ...)

Usage:
    prescriber = RAGPrescriber(database_path=Path("..."), hierarchy_builder=build_ppu_hierarchy)

    # Retrieve best matching pattern for an optimization hint
    opt_node, similarity_scores = prescriber.retrieve("use AIU load for memory optimization")

    # Build context from the retrieved node (traverses to leaf code examples)
    context = prescriber.build_context(opt_node)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np

from kernel_perf_agent.kernel_opt.database.base import OptHierarchy, OptNode


def _keyword_tokens(text: str) -> set[str]:
    """Return a cheap keyword set for fallback retrieval."""
    return set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))


class RAGPrescriber:
    """
    Embedding-based retriever for optimization patterns.

    Uses OpenAI-compatible embeddings to find the most similar optimization node
    based on the optimization prompt, then traverses to leaf nodes to collect
    code examples. Falls back to keyword-overlap matching when embeddings are
    unavailable so retrieval never hard-fails.
    """

    def __init__(
        self,
        database_path: Path | None = None,
        logger: logging.Logger | None = None,
        hierarchy_builder: Callable[[Path], OptHierarchy] | None = None,
    ):
        """
        Initialize the RAG prescriber.

        Args:
            database_path: Path to the code_samples directory. Defaults to
                           kernel_perf_agent/kernel_opt/database/code_samples
            logger: Optional logger instance
            hierarchy_builder: Optional callable that builds an OptHierarchy from
                               ``database_path``. Defaults to the bundled NVIDIA
                               ``OptHierarchy.hard_initialize``; PPU passes
                               ``build_ppu_hierarchy``.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.opt_hierarchy: OptHierarchy | None = None
        self._openai_client = None  # Lazy-loaded
        self._embeddings_available = True
        self._node_embeddings: dict[OptNode, list[float]] = {}  # L1/L2 embeddings

        if database_path is None:
            project_root = Path(__file__).resolve()
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            database_path = (
                project_root
                / "kernel_perf_agent"
                / "kernel_opt"
                / "database"
                / "code_samples"
            )

        self._initialize_database(database_path, hierarchy_builder)

    def _initialize_database(
        self,
        db_path: Path,
        hierarchy_builder: Callable[[Path], OptHierarchy] | None,
    ) -> None:
        """Initialize the optimization hierarchy from disk."""
        if not db_path.exists():
            self.logger.warning(f"RAG database path not found: {db_path}")
            return

        if hierarchy_builder is not None:
            self.opt_hierarchy = hierarchy_builder(db_path)
        else:
            self.opt_hierarchy = OptHierarchy()
            self.opt_hierarchy.hard_initialize(db_path)

        self._precompute_embeddings()
        self.logger.info(f"Initialized RAG database from {db_path}")

    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for L1/L2 nodes (best effort)."""
        if not self.opt_hierarchy:
            return

        root = self.opt_hierarchy.get_root()
        cur_level = list(root.opt_children)

        while cur_level:
            child_level = []
            for node in cur_level:
                if node.opt_children:  # only L1/L2 (non-leaf) nodes
                    try:
                        self._node_embeddings[node] = self._embed_query(node.opt_desc)
                    except Exception as exc:
                        self._embeddings_available = False
                        self.logger.warning(
                            "Embedding failed (%s); will use keyword fallback", exc
                        )
                        self._node_embeddings.clear()
                        return
                    for child in node.opt_children:
                        if child not in child_level:
                            child_level.append(child)
            cur_level = child_level

        if self._node_embeddings:
            self.logger.info(
                f"Precomputed embeddings for {len(self._node_embeddings)} L1/L2 nodes"
            )

    def _get_openai_client(self):
        """Lazy-load an OpenAI-compatible client.

        Reads OPENAI_API_KEY/OPENAI_BASE_URL first, then falls back to the
        ADVIBE_API_KEY/ADVIBE_BASE_URL gateway env vars used by the rest of the
        project, so the same embeddings gateway can serve both.
        """
        if self._openai_client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ADVIBE_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("ADVIBE_BASE_URL")
            kwargs: dict[str, str] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def _embed_query(self, text: str) -> list[float]:
        """Get embedding for a text query."""
        client = self._get_openai_client()
        model = os.getenv("KERNELAGENT_EMBEDDING_MODEL", "text-embedding-3-large")
        response = client.embeddings.create(
            input=text,
            model=model,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1, vec2) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def _keyword_similarity(self, query: str, text: str) -> float:
        q = _keyword_tokens(query)
        t = _keyword_tokens(text)
        if not q or not t:
            return 0.0
        return len(q & t) / len(q | t)

    def _candidate_nodes(self) -> list[OptNode]:
        """Return all non-leaf (L1/L2) nodes for retrieval."""
        if not self.opt_hierarchy:
            return []
        nodes: list[OptNode] = []
        stack = list(self.opt_hierarchy.get_root().opt_children)
        while stack:
            node = stack.pop()
            if node.opt_children:
                nodes.append(node)
                stack.extend(node.opt_children)
        return nodes

    def retrieve(self, opt_prompt: str) -> tuple[OptNode | None, dict[OptNode, float]]:
        """
        Retrieve the most relevant optimization node.

        Uses embedding similarity when available, otherwise keyword overlap.
        """
        if not self.opt_hierarchy:
            self.logger.warning("Database not initialized")
            return None, {}

        opt_similarity: dict[OptNode, float] = {}

        if self._node_embeddings:
            try:
                key_embedding = self._embed_query(opt_prompt)
            except Exception as exc:
                self.logger.warning("Query embedding failed (%s); falling back", exc)
                self._node_embeddings = {}
            else:
                for node, node_embedding in self._node_embeddings.items():
                    opt_similarity[node] = self._cosine_similarity(
                        key_embedding, node_embedding
                    )

        if not opt_similarity:
            # Keyword fallback over all L1/L2 nodes.
            for node in self._candidate_nodes():
                opt_similarity[node] = self._keyword_similarity(
                    opt_prompt, node.opt_desc
                )

        if not opt_similarity:
            return None, {}

        best_node = max(opt_similarity, key=opt_similarity.get)
        self.logger.info(
            f"Retrieved optimization pattern (score={opt_similarity[best_node]:.3f}): "
            f"{best_node.opt_desc.strip()[:80]}..."
        )
        return best_node, opt_similarity

    def build_context(
        self,
        opt_node: OptNode,
        max_chars: int = 8192,
        max_code_examples: int = 2,
    ) -> str:
        """
        Build context from opt_node with technique description and code examples.
        """
        parts: list[str] = []
        code_examples: list[str] = []

        cur_level = [opt_node]
        while cur_level:
            child_level = []
            for node in cur_level:
                if node.opt_children:
                    parts.append(node.opt_desc.strip())
                else:
                    code_examples.append(node.opt_desc.strip())
                for child in node.opt_children:
                    if child not in child_level:
                        child_level.append(child)
            cur_level = child_level

        context = ""
        if parts:
            context += "## Optimization Technique\n\n" + "\n\n---\n\n".join(parts)
        if code_examples:
            selected = code_examples[:max_code_examples]
            if parts:
                context += "\n\n---\n\n"
            context += "## Code Examples\n\n" + "\n\n---\n\n".join(selected)
            if len(code_examples) > max_code_examples:
                context += f"\n\n(Showing {max_code_examples} of {len(code_examples)} examples)"

        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n... (truncated)"
            self.logger.warning(f"Context truncated to {max_chars} chars")

        return context
