from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree

############################## GENERAL SETTINGS ##############################

# Level of logging output
# Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = True

# Number of processes used to run simulations in parallel.
# This option is ignored if PARALLEL_EXECUTION = False
N_PROCESSES = cpu_count()

# Format in which results are saved.
# Result readers and writers are located in module ./icarus/results/readwrite.py
# Currently only PICKLE is supported
RESULTS_FORMAT = 'PICKLE'

# Number of times each experiment is replicated
# This is necessary for extracting confidence interval of selected metrics
N_REPLICATIONS = 1

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icaurs/execution/collectors.py
# Remove collectors not needed
DATA_COLLECTORS = [
           'CACHE_HIT_RATIO',  # Measure cache hit ratio
           'LATENCY',  # Measure request and response latency (based on static link delays)
           'OVERHEAD_DISTRIBUTION', # Coefficient of Variation on Cache Hits
           'CACHE_EVICTION', # Average number of evictions per uCache
           'CACHING_EFFICIENCY',
           'SEGMENT_PERFORMANCE_DIFFERENCE', 
                   ]



########################## EXPERIMENTS CONFIGURATION ##########################

# Default experiment values, i.e. values shared by all experiments

# Number of content objects
N_CONTENTS = 1

# Numer of segments per content object
N_SEGMENTS = 5000

TIME_INTERVALS = [1
                  ]

N_CONTENTS = N_CONTENTS * N_SEGMENTS

# Number of content requests generated to pre-populate the caches
# These requests are not logged
N_WARMUP_REQUESTS = 250

# Number of content requests that are measured after warmup
N_MEASURED_REQUESTS = 750

# Number of requests per second (over the whole network)
REQ_RATE = 10.0

# Cache eviction policy
CACHE_POLICY = 'LRU'

# Zipf alpha parameter, remove parameters not needed
ALPHA = [1.0]

# Total size of network cache as a fraction of content population
# Remove sizes not needed
NETWORK_CACHE = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Total cache budget
# cache_budget = [nCache_budget, uCache_budget]
CACHE_BUDGET_FACTORS = [ # [1.0, 0], # nCache only
                         # [1.0, 1.0], # Both nCache and uCache
                         [0, 1.0], # uCache only
                ]

# List of topologies tested
# Topology implementations are located in ./icarus/scenarios/topology.py
# Remove topologies not needed
TOPOLOGIES = [
	# 'TREE_WITH_UCACHE'
  'ROCKET_FUEL_WITH_UCACHE',
]

# List of caching and routing strategies
# The code is located in ./icarus/models/strategy/*.py
# Remove strategies not needed
STRATEGIES = [
    # 'LCE',                 # Leave Copy Everywhere
    # 'LCE_USER_ASSISTED',   # Leave Copy Everywhere User-Assisted
    # 'C_RANDOM',
    'C_LFR_UM',           # Centralised Largest Future Request First with User-Matching
    'C_LCF_UM',           # Centralised Least Cached First with User-Matching
    'C_RANDOM_UM',        # Centralised Random with User-Matching
]

# Instantiate experiment queue
EXPERIMENT_QUEUE = deque()

# Build a default experiment configuration which is going to be used by all
# experiments of the campaign
default = Tree()
default['workload'] = {'name':          'STATIONARY',
                       'n_contents':    N_CONTENTS,
                       'n_segments':    N_SEGMENTS,
                       'n_warmup':      N_WARMUP_REQUESTS,
                       'n_measured':    N_MEASURED_REQUESTS,
                       'rate':          REQ_RATE}

default['cache_placement']['name'] = 'UNIFORM_WITH_UCACHE'

default['content_placement'] = {'name':       'UNIFORM',
                                'n_contents': N_CONTENTS,
                                'n_segments': N_SEGMENTS}

default['cache_policy']['name'] = CACHE_POLICY

default['topology'] = {'asn':  3257,  # Tiscali (Europe)
                      }

# Create experiments multiplexing all desired parameters
for alpha in ALPHA:
    for strategy in STRATEGIES:
        for topology in TOPOLOGIES:
          for time_interval in TIME_INTERVALS:
            for cache_budget_factor in CACHE_BUDGET_FACTORS:
              for network_cache in NETWORK_CACHE:
                  experiment = copy.deepcopy(default)
                  experiment['workload']['alpha'] = alpha
                  experiment['workload']['time_interval'] = time_interval
                  experiment['strategy']['name'] = strategy
                  experiment['topology']['name'] = topology
                  experiment['cache_placement']['nCache_budget'] = cache_budget_factor[0] * N_CONTENTS * 161 * 0.1
                  experiment['cache_placement']['uCache_budget'] = cache_budget_factor[1] * N_CONTENTS * 161 * network_cache
                  experiment['cache_placement']['network_cache'] = network_cache
                  experiment['desc'] = "strategy: %s, time interval: %s, network cache: %s" \
                                       % (strategy, str(time_interval), str(network_cache))
                  EXPERIMENT_QUEUE.append(experiment)
