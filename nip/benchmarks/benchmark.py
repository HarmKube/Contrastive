from ultralytics.utils.benchmarks import benchmark

def benchmark_model(model: str, data: str, imgsz: int = 1920, device: str = 'cuda'):
    """
    Benchmark a YOLOv8 model on a dataset. This tests the different export formats and reports fps.

    Args:
        model (str): Path to the YOLOv8 model file.
        data (str): Path to the dataset configuration file.
        imgsz (int): Image size for the benchmark. Default is 1920.
        device (str): Device to run the benchmark on. Default is 'cuda'.
    """
    benchmark(model=model, data=data, imgsz=imgsz, device=device)