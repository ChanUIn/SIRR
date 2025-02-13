#compute.py
import time
import torch
from thop import profile
from TSmodel import T_teacher_net, R_teacher_net, student_net


def compute_gflops_and_model_size(model, input_args):
    """
    Computes the GFLOPs and model size (MB) of a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_args (tuple): The inputs for the model.

    Returns:
        tuple: (params_M, model_size, GFlops)
            - params_M: Number of parameters in millions.
            - model_size: Model size in megabytes.
            - GFlops: Floating point operations in GFLOPs.
    """
    macs, params = profile(model, inputs=input_args, verbose=False)

    GFlops = macs * 2.0 / 1e9  #  GFLOPs pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024  # to MB
    params_M = params / pow(10, 6)  # millions
    return params_M, model_size, GFlops


@torch.no_grad()
def compute_fps(model, input_args, device=None, iterations=3):
    """
    Computes the average FPS (frames per second) of a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_args (tuple): The inputs for the model.
        epoch (int): Number of iterations to compute the average.
        device (torch.device): Device to perform computation (e.g., 'cuda:0').

    Returns:
        tuple: (fps, runtime)
            - fps: Frames per second.
            - runtime: Total time in seconds to evaluate `epoch` iterations.
    """
    total_time = 0.0

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start = time.time()
    for _ in range(iterations):
        outputs = model(*input_args)
    end = time.time()
    total_time += (end - start)

    #avg_time_per_frame = total_time / epoch
    #fps = 1 / avg_time_per_frame
    avg_time_per_iteration = (end - start) / iterations
    fps = 1 / avg_time_per_iteration
    return fps, avg_time_per_iteration


def evaluate_model(model, model_name, input_args):
    """
    Computes and returns the metrics for a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_name (str): Name of the model.
        input_args (tuple): The inputs for the model.

    Returns:
        tuple: (str, tuple) The formatted string and raw numerical metrics.
    """
    start_time = time.time()
    params_M, model_size, gflops = compute_gflops_and_model_size(model, input_args)
    fps, runtime = compute_fps(model, input_args, iterations=3)
    end_time = time.time()
    total_runtime = end_time - start_time

    formatted_metrics = f"=== {model_name} Metrics ===\n" \
                        f"Model Parameters: {params_M:.2f}M (Million)\n" \
                        f"Model Size: {model_size:.2f} MB\n" \
                        f"GFLOPs: {gflops:.2f} GFLOPs\n" \
                        f"FPS: {fps:.2f} Frames Per Second\n" \
                        f"Runtime: {runtime:.2f} seconds (for iterations)\n" \
                        f"Run Time: {total_runtime:.2f} seconds\n"

    return formatted_metrics, (params_M, model_size, gflops, fps, runtime, total_runtime)

def main():
    """Computes metrics for T_teacher_net, R_teacher_net, and student_net, and writes results to a file."""
    output_path = r"E:\CHAN\absorption_2\main\TS\ana.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define c before using it
    c = torch.tensor([0.5], dtype=torch.float32).to(device)

    input_t = torch.randn(1, 3, 256, 256).to(device)
    input_r = input_t
    input_s = (torch.randn(1, 3, 256, 256).to(device),)

    t_teacher = T_teacher_net().to(device)
    t_teacher_metrics, t_teacher_values = evaluate_model(t_teacher, "T_teacher_net", (input_t, c))

    r_teacher = R_teacher_net().to(device)
    r_teacher_metrics, r_teacher_values = evaluate_model(r_teacher, "R_teacher_net", (input_r, c))

    student = student_net().to(device)
    input_student = (input_s[0], c)
    student_metrics, student_values = evaluate_model(student, "student_net", input_student)

    total_params = t_teacher_values[0] + r_teacher_values[0] + student_values[0]
    total_size = t_teacher_values[1] + r_teacher_values[1] + student_values[1]
    total_gflops = t_teacher_values[2] + r_teacher_values[2] + student_values[2]
    total_fps = t_teacher_values[3] + r_teacher_values[3] + student_values[3]
    total_runtime = t_teacher_values[4] + r_teacher_values[4] + student_values[4]
    total_evlruntime = t_teacher_values[5] + r_teacher_values[5] + student_values[5]

    total_metrics = f"=== Total TSModel Metrics ===\n" \
                    f"Total Parameters: {total_params:.2f}M (Million)\n" \
                    f"Total Model Size: {total_size:.2f} MB\n" \
                    f"Total GFLOPs: {total_gflops:.2f} GFLOPs\n" \
                    f"Total FPS: {total_fps:.2f} Frames Per Second\n" \
                    f"Total run_time: {total_runtime:.2f} seconds\n" \
                    f"Total Evaluation Time: {total_evlruntime:.2f} seconds\n"

    all_metrics = f"{t_teacher_metrics}\n{r_teacher_metrics}\n{student_metrics}\n{total_metrics}"
    print(all_metrics)

    with open(output_path, "w") as f:
        f.write(all_metrics)

if __name__ == '__main__':
    main()
