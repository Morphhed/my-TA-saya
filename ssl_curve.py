import csv
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import TENSORBOARD_DIR, OUT_42, WARMUP, TEMPERATURE

def read_scalar(event_accumulator, tag):
    events = event_accumulator.Scalars(tag)
    values = {}

    for event in events:
        values[event.step] = event.value

    return sorted(values.items())

def main():
    if not TENSORBOARD_DIR.exists():
        raise FileNotFoundError(f"TensorBoard directory tidak ditemukan: {TENSORBOARD_DIR}")

    ea = EventAccumulator(str(TENSORBOARD_DIR), size_guidance={"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])

    train_tag = "Training/Epoch_Loss"
    val_tag = "Validation/NTXent_Loss"

    if train_tag not in available_tags or val_tag not in available_tags:
        raise RuntimeError(f"Tag tidak lengkap. Ditemukan: {available_tags}")

    train_data = read_scalar(ea, train_tag)
    val_data = read_scalar(ea, val_tag)

    train_epochs = [step for step, value in train_data]
    train_losses = [value for step, value in train_data]

    val_epochs = [step for step, value in val_data]
    val_losses = [value for step, value in val_data]

    best_index = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    best_epoch = val_epochs[best_index]
    best_val_loss = val_losses[best_index]

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label="Training NT-Xent Loss")
    plt.plot(val_epochs, val_losses, label="Validation NT-Xent Loss")
    plt.axvline(WARMUP, linestyle="--", label=f"End Warm-up ({WARMUP})")
    plt.scatter([best_epoch], [best_val_loss], zorder=5, label=f"Best Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title(f"SimCLR Training Curve - Temperature {TEMPERATURE}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plot_path = OUT_42 / "4_2_simclr_loss_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = OUT_42 / "4_2_simclr_loss.csv"

    train_dict = dict(train_data)
    val_dict = dict(val_data)
    all_epochs = sorted(set(train_dict) | set(val_dict))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "validation_loss"])

        for epoch in all_epochs:
            writer.writerow([epoch, train_dict.get(epoch, ""), val_dict.get(epoch, "")])

    summary_path = OUT_42 / "4_2_ssl_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SIMCLR TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Temperature       : {TEMPERATURE}\n")
        f.write(f"Best Epoch        : {best_epoch}\n")
        f.write(f"Best Val NT-Xent  : {best_val_loss:.6f}\n")
        f.write(f"Warm-up Epoch     : {WARMUP}\n")

    print(f"[Saved] {plot_path}")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()