from benchmarks.benchmark import benchmark_model
from cli.cli_models import CliArgument, CliAction
from dataset.video import create_video_dataset, parse_playlist
from benchmarks.val import validate_yolo
from interpretation.model_interpret import interpret_global, interpret_local
from sslcon.train import train
from kfolds.kfolds import kfolds
from dataset.create_labels import create_labels_from_yolo
from dataset.analyse import analyse_dataset, compare_analysis
from visualization.visualize import visualize_annotation


argument: list[CliArgument] = [
    ### FLAGS ###
    CliArgument('cli-test', True, 'ct', 'Test the CLI'),
    CliArgument('benchmark', True, 'b', 'Run a benchmark on the model'),
    CliArgument('validate', True, 'v', 'Run a test run on the model'),
    CliArgument('train', True, 't', 'Train the model'),
    CliArgument('video-dataset', True, 'vd', 'Create a dataset from a video (YouTube URL or local path)'),
    CliArgument('create-labels', True, 'cl', 'Create labels from a YOLO model'),
    CliArgument('visualize-annotation', True, 'va', 'Visualize an Annotation'),
    CliArgument('analyse-dataset', True, 'ad', 'Analyse a dataset'),
    CliArgument('parse_playlist', True, 'pp', 'Parse a YouTube playlistand writes the video URLs to a file csv file'),
    CliArgument('compare-analysis', True, 'ca', 'Compare two analysis files'),
    CliArgument('lightly-train', True, 'lt', 'Train a model using lightly'),
    CliArgument('interpret-global', True, 'ig', 'Interpret a models backbone using Grad-CAM'),
    CliArgument('interpret-local', True, 'il', 'Interpret a models output using Grad-CAM'),
    CliArgument('kfolds', True, 'kf', 'Train the model on kfolds of the dataset'),

    ### COMMON ARGUMENTS ###
    CliArgument('model', False, 'm', 'Path to the model file'),
    CliArgument('data', False, 'd', 'Path to the dataset configuration file'),
    CliArgument('imgsz', False, 'i', 'Image size'),
    CliArgument('device', False, 'dev', 'Device to run the script on.'),
    CliArgument('output', False, 'o', 'Output directory'),
    CliArgument('batch_size', False, 'bs', 'Batch size for processing images'),
    CliArgument('config', False, 'c', 'Path to the config yaml to use'),
    CliArgument('indices', False, 'idx', 'Indices of images to process, comma-separated (e.g., 0,1,2)'),

    # Training arguments MoCo #
    CliArgument('ft_coco', True, 'fc', 'Fine-Tune on Coco after pretraining.'),
    CliArgument('local_data', False, 'ld', 'Specify path to pretrain data.'),
    CliArgument('moco_ckpt', False, 'mc', 'Specify path to a lightning checkpoint.'),
    CliArgument('yolo_ckpt', False, 'yc', 'Specify path to a yolo pt-File.'),
    CliArgument('lightly_ckpt', True, 'lc', 'Specify if lightly should resume training.'),
    CliArgument('ssl_mode', False, 'sm', 'Pretraining using the framework. Valid options are: YOLOMoCo, YOLODetCo, lightly'),
    CliArgument('split', False, 'split', 'Split to analyse'),

    # Video dataset arguments
    CliArgument('source', False, 's', 'YouTube URL or local video file path'),
    CliArgument('interval', False, 'int', 'Time interval between frames in seconds'),
    CliArgument('resolution', False, 'res', 'Video resolution for YouTube downloads (e.g., 720p)'),
    CliArgument('output_filename', False, 'ofn', 'Desired filename of the video, only used if source is a YouTube URL'),

    # Create labels arguments
    CliArgument('confidence_threshold', False, 'conf', 'Confidence threshold for detections'),

    # Visualize annotation arguments
    CliArgument('xml_path', False, 'xml', 'Path to the XML annotation file'),

    # Dataset comparison arguments
    CliArgument('file_1', False, 'f1', 'Path to the first analysis file'),
    CliArgument('file_2', False, 'f2', 'Path to the second analysis file'),
    # Extend this list with other arguments as needed. Pay attention if your argument already exists in the list and dont use a short name that is already used.

    # kfold arguments
    CliArgument('kfold_iter', False, 'ki', 'Specify kfold iteration, that we want to continue from.'),
    CliArgument('kfolds_val', True, 'kv', 'Validate on k-Fold Validation Folds'),
]

actions: list[CliAction] = [
    CliAction('cli-test', 'cli_test', [],[], lambda : print(f"Cli is working!")),
    CliAction('benchmark', 'benchmark', ['model', 'data'], ['imgsz', 'device'],benchmark_model), 
    CliAction('train', 'train', ['data'], ['imgsz', 'output', 'ft_coco', 'ssl_mode', 'local_data', 'moco_ckpt', 'yolo_ckpt', 'lightly_ckpt'],train),
    CliAction('kfolds', 'kfolds', ['data'], ['imgsz', 'output', 'yolo_ckpt', 'kfold_iter'], kfolds),
    CliAction('validate', 'validate', ['output', 'data'], ['imgsz', 'device', 'kfolds_val'],validate_yolo),
    # Video dataset action
    CliAction('video-dataset', 'video_dataset', ['source', 'output'], ['interval', 'resolution', 'output_filename'], 
              create_video_dataset),
    CliAction('parse_playlist', 'parse_playlist', ['source', 'output'], [], parse_playlist),
    CliAction('create-labels', 'create_labels', ['data', 'model',] , ['output','confidence_threshold', 'device', 'batch_size'], create_labels_from_yolo),
    CliAction('visualize-annotation', 'visualize_annotation', ['xml_path'], ['output'], visualize_annotation),
    CliAction('analyse-dataset', 'analyse_dataset', ['data'], ['split'], lambda data, split = '': analyse_dataset(data,split)),
    CliAction('compare-analysis', 'compare_analysis', ["file_1", 'file_2'], [], compare_analysis),
    CliAction('interpret-global', 'interpret_global', ['model', 'output', 'data', 'indices'], [], 
              interpret_global),
              CliAction('interpret-local', 'interpret_local', ['model', 'output', 'data', 'indices'], [], 
              interpret_local),
    # Extend this list with other actions as needed.
]