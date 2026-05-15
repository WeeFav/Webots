from datetime import datetime

# DATASET
dataset = 'Carla'
data_root = 'C:\\Users\\marvi\\Datasets\\Lane\\CarlaLane'
num_lanes = 4
num_cls = 4

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True
use_classification = True

# TRAIN
epoch = 50
batch_size = 16
optimizer = 'Adam' # ['SGD','Adam']
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

# SCHEDULER
scheduler = 'cos' # ['multi', 'cos']
steps = [25,38] # Not used in cosine schedule
gamma  = 0.1 # Ignored in cosine, only used in step-based scheduler
warmup = 'linear'
warmup_iters = 100  

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None
timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = f"C:\\Users\\marvi\\carla_lane_detection\\weights\\{timestamp}"