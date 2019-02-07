"""Implementations of all on-path strategies"""
from __future__ import division
import random

import networkx as nx

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
       'Centralised_LeastCachedFirst_UM', # Last Modified: 2018.12.13
       'Centralised_LargestFutureRequestFirst_UM', # Last Modified: 2018.12.13
       'Centralised_Random', # Last Modified: 2018.12.05
       'Centralised_Random_UM', # Last Modified: 2018.12.05
       'Partition',
       'Edge',
       'LeaveCopyEverywhere',
       'LeaveCopyEverywhere_UserAssisted', # Last Modified: 2018.10.29
       'LeaveCopyDown',
       'ProbCache',
       'CacheLessForMore',
       'RandomBernoulli',
       'RandomChoice',
           ]

@register_strategy('C_LCF_UM')
class Centralised_LeastCachedFirst_UM(Strategy):

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Centralised_LeastCachedFirst_UM, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, n_segments, time_interval, log):
        self.controller.update_user_cache_table(time, time_interval)
        # Start session.
        self.controller.start_session(time, receiver, content, log)
        # Check if the receiver has already cached the content.
        if self.view.has_cache(receiver):
            if self.controller.get_content(receiver):
                # If the segment is the last segment, cache all the previous segments starting from segment with least cache counts.
                if content % n_segments == 0:
                    self.controller.end_session()
                    segments = self.controller.sort_by_LCF(range(content-n_segments+1, content+1))
                    for segment in segments:
                        self.controller.start_session(time, receiver, segment, log)
                        self.controller.put_content(receiver)
                        self.controller.end_session()
                    return None
            else:
                self.controller.end_session()
                return None
        # Receiver does not cache the content, get all required data.
        content_locations = list(self.view.content_locations(content))
        # print ("Content locations: " + str(content_locations) + " for " + str(segment))
        destination = content_locations[0]
        path = self.view.shortest_path(receiver, destination)
        # Find the nearest content location and the corresponding shortest path.
        for content_location in content_locations:
            current_path = self.view.shortest_path(receiver, content_location)
            if len(current_path) < len(path):
                path = current_path
                destination = content_location
        # Route request to destination.
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
        # Get content from destination.
        self.controller.get_content(destination)
        # Route content to receiver.
        path = list(reversed(path))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        # If the segment is the last segment, cache all the previous segments starting from segment with least cache counts.
        if content % n_segments == 0:
            self.controller.end_session()
            segments = self.controller.sort_by_LCF(range(content-n_segments+1, content+1))
            for segment in segments:
                self.controller.start_session(time, receiver, segment, log)
                self.controller.put_content(receiver)
                self.controller.end_session()
            return None
        else:
            self.controller.end_session()
            return None

@register_strategy('C_LFR_UM')
class Centralised_LargestFutureRequestFirst_UM(Strategy):

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Centralised_LargestFutureRequestFirst_UM, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, n_segments, time_interval, log):
        self.controller.update_user_download_table(time, time_interval)
        # Start session.
        self.controller.start_session(time, receiver, content, log)
        # Check if the receiver has already cached the content.
        if self.view.has_cache(receiver):
            if self.controller.get_content(receiver):
                # If the segment is the last segment, cache all the previous segments starting from segment with least download counts.
                if content % n_segments == 0:
                    self.controller.end_session()
                    segments = self.controller.sort_by_LFR(range(content-n_segments+1, content+1))
                    for segment in segments:
                        self.controller.start_session(time, receiver, segment, log)
                        self.controller.put_content(receiver)
                        self.controller.end_session()
                    return None
                else:
                    self.controller.end_session()
                    return None
        # Receiver does not cache the content, get all required data.
        content_locations = list(self.view.content_locations(content))
        # print ("Content locations: " + str(content_locations) + " for " + str(segment))
        destination = content_locations[0]
        path = self.view.shortest_path(receiver, destination)
        # Find the nearest content location and the corresponding shortest path.
        for content_location in content_locations:
            current_path = self.view.shortest_path(receiver, content_location)
            if len(current_path) < len(path):
                path = current_path
                destination = content_location
        # Route request to destination.
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
        # Get content from destination.
        self.controller.get_content(destination)
        # Route content to receiver.
        path = list(reversed(path))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        # If the segment is the last segment, cache all the previous segments starting from segment with least download counts.
        if content % n_segments == 0:
            self.controller.end_session()
            segments = self.controller.sort_by_LFR(range(content-n_segments+1, content+1))
            for segment in segments:
                self.controller.start_session(time, receiver, segment, log)
                self.controller.put_content(receiver)
                self.controller.end_session()
            return None
        else:
            self.controller.end_session()
            return None

@register_strategy('C_RANDOM_UM')
class Centralised_Random_UM(Strategy):

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Centralised_Random_UM, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, n_segments, time_interval, log):
        # Start session.
        self.controller.start_session(time, receiver, content, log)
        # Check if the receiver has already cached the content.
        if self.view.has_cache(receiver):
            if self.controller.get_content(receiver):
                # If the segment is the last segment, cache all the previous segments in a random order.
                if content % n_segments == 0:
                    self.controller.end_session()
                    segments = range(content-n_segments+1, content+1)
                    random.shuffle(segments)
                    for segment in segments:
                        self.controller.start_session(time, receiver, segment, log)
                        self.controller.put_content(receiver)
                        self.controller.end_session()
                    return None
                else:
                    self.controller.end_session()
                    return None

        # Receiver does not cache the content, get all required data.
        content_locations = list(self.view.content_locations(content))
        # print ("Content locations: " + str(content_locations) + " for " + str(segment))
        destination = content_locations[0]
        path = self.view.shortest_path(receiver, destination)
        # Find the nearest content location and the corresponding shortest path.
        for content_location in content_locations:
            current_path = self.view.shortest_path(receiver, content_location)
            if len(current_path) < len(path):
                path = current_path
                destination = content_location
        # Route request to destination.
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
        # Get content from destination.
        self.controller.get_content(destination)
        # Route content to receiver.
        path = list(reversed(path))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        # If the segment is the last segment, cache all the previous segments in a random order.
        if content % n_segments == 0:
            self.controller.end_session()
            segments = range(content-n_segments+1, content+1)
            random.shuffle(segments)
            for segment in segments:
                self.controller.start_session(time, receiver, segment, log)
                self.controller.put_content(receiver)
                self.controller.end_session()
        else:
            self.controller.end_session()

@register_strategy('C_RANDOM')
class Centralised_Random(Strategy):

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(Centralised_Random, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # Start session
        self.controller.start_session(time, receiver, content, log)
        # Check if the receiver has already cached the content, if true, end the session.
        if self.view.has_cache(receiver):
            if self.controller.get_content(receiver):
                self.controller.end_session()
                return None
        # Receiver does not cache the content, get all required data.
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route request to source.
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
        # Get content from source.
        self.controller.get_content(source)
        # Route content to receiver.
        path = list(reversed(path))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
        # If receiver's cache is full, evict a content by random.
        if self.view.is_cache_full(receiver):
            self.controller.remove_content_by_random(receiver)
        # Insert content.
        self.controller.put_content(receiver)
        # End session
        self.controller.end_session()

@register_strategy('PARTITION')
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Partition, self).__init__(view, controller)
        if 'cache_assignment' not in self.view.topology().graph:
            raise ValueError('The topology does not have cache assignment '
                             'information. Have you used the optimal median '
                             'cache assignment?')
        self.cache_assignment = self.view.topology().graph['cache_assignment']

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy('EDGE')
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Edge, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


@register_strategy('LCE')
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('LCE_USER_ASSISTED')
class LeaveCopyEverywhere_UserAssisted(Strategy):
    """Leave Copy Everywhere (LCE) user-assisted strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between receiver (inclusive) and serving node.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere_UserAssisted, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # Start session.
        self.controller.start_session(time, receiver, content, log)
        # Check if the receiver has already cached the content, if true, end the session.
        if self.view.has_cache(receiver):
            if self.controller.get_content(receiver):
                self.controller.end_session()
                return None
        # Receiver does not cache the content, get all required data.
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route request to original source and queries caches on the path.
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source.
            self.controller.get_content(v)
            serving_node = v
        # Route content to receiver.
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # Insert content.
                self.controller.put_content(v)
        # End session.
        self.controller.end_session()

@register_strategy('LCD')
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()


@register_strategy('PROB_CACHE')
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(ProbCache, self).__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)

        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([v for v in path if self.view.has_cache(v)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum([self.cache_size[n] for n in path[hop - 1:]
                     if n in self.cache_size])
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CL4M')
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(CacheLessForMore, self).__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(topology, v))[v])
                             for v in topology.nodes_iter())
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_BERNOULLI')
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super(RandomBernoulli, self).__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
        self.controller.end_session()

@register_strategy('RAND_CHOICE')
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(RandomChoice, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()
