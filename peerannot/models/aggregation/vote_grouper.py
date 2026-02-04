from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
from networkx.algorithms.community import louvain_communities


class VoteGrouper:
    @staticmethod
    def build_counts(all_votes):
        """Build symmetric confusion counts from votes."""
        counts = defaultdict(lambda: defaultdict(int))
        for votes in all_votes.values():
            labels = list(votes.values())
            for a, b in combinations(labels, 2):
                counts[a][b] += 1
                counts[b][a] += 1
        return counts

    @staticmethod
    def normalize_rows(confusion_counts):
        """Convert counts into row-normalized probabilities."""
        probs = {}
        for label, row in confusion_counts.items():
            total = sum(row.values())
            if total > 0:
                probs[label] = {k: v / total for k, v in row.items()}
        return probs

    @staticmethod
    def build_graph(confusion_probs, threshold=0.05):
        """Construct a weighted undirected graph from confusion probabilities."""
        G = nx.Graph()
        seen = set()

        for a, row in confusion_probs.items():
            for b, pab in row.items():
                # will prevent double counting
                if (a, b) in seen or (b, a) in seen:
                    continue
                seen.add((a, b))

                pba = confusion_probs.get(b, {}).get(a, 0.0)
                weight = pab + pba
                if weight >= threshold:
                    G.add_edge(a, b, weight=weight)

        return G

    @staticmethod
    def detect_blocks(confusion_probs, graph):
        """Run Louvain community detection and assign blocks."""
        communities = louvain_communities(graph, weight="weight")
        blocks = {cls: i for i, comm in enumerate(communities) for cls in comm}

        # assign any remaining nodes to a "weak" group
        weak_group_id = len(communities)
        for node in confusion_probs.keys():
            if node not in blocks:
                blocks[node] = weak_group_id

        return blocks

    @staticmethod
    def group_votes_by_blocks(all_votes, blocks):
        """Organize tasks by the dominant block of their labels."""
        grouped = defaultdict(dict)
        for task_id, votes in all_votes.items():
            labels = list(votes.values())
            block_counts = Counter(blocks[l] for l in labels if l in blocks)

            if not block_counts:
                continue

            dominant_block = block_counts.most_common(1)[0][0]
            grouped[dominant_block][task_id] = votes

        return dict(grouped)

    @staticmethod
    def group_votes(all_votes, threshold=0.05):
        """returns grouped votes"""
        counts = VoteGrouper.build_counts(all_votes)
        probs = VoteGrouper.normalize_rows(counts)
        graph = VoteGrouper.build_graph(probs, threshold)
        blocks = VoteGrouper.detect_blocks(probs, graph)
        mini_batches = VoteGrouper.group_votes_by_blocks(all_votes, blocks)
        return mini_batches
