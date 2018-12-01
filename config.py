# parameters for processing the dataset
DATA_PATH = 'data/'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'train.en'
OUTPUT_FILE = 'train.vi'
PROCESSED_PATH = 'data'
CPT_PATH = 'checkpoints'

ENC_VOCAB = 41303

DEC_VOCAB = 18778

THRESHOLD = 1

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 256

LR = 0.5
MAX_GRAD_NORM = 5.0

# NUM_SAMPLES = 512
NUM_SAMPLES = 18777

MAX_ITERATION = 30001