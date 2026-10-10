"""Create an SVG chart from the 20 recorded end-of-epoch training log values.

The values are the final displayed mini-batch loss for each epoch in
notebooks/Transformer_Colab_Training_and_Inference.ipynb. They are not
epoch averages and are not validation metrics.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


LOSSES = [
    4.051, 3.407, 3.248, 2.988, 2.823, 2.866, 3.399, 2.981, 2.752, 2.571,
    2.491, 2.351, 2.331, 2.225, 2.288, 2.600, 2.332, 2.259, 2.124, 2.086,
]


def main() -> None:
    output = ROOT / "docs" / "assets" / "recorded_training_loss.svg"
    png_output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, left, right, top, bottom = 1080, 560, 90, 35, 70, 100
    chart_width, chart_height = width - left - right, height - top - bottom
    minimum, maximum = 1.0, 4.5

    def x(epoch: int) -> float:
        return left + epoch * chart_width / (len(LOSSES) - 1)

    def y(loss: float) -> float:
        return top + (maximum - loss) * chart_height / (maximum - minimum)

    points = " ".join(f"{x(epoch):.1f},{y(loss):.1f}" for epoch, loss in enumerate(LOSSES))
    horizontal = []
    for value in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        horizontal.append(f'<line x1="{left}" y1="{y(value):.1f}" x2="{width-right}" y2="{y(value):.1f}" class="grid"/>')
        horizontal.append(f'<text x="{left-12}" y="{y(value)+5:.1f}" text-anchor="end" class="axis">{value:.1f}</text>')
    ticks = "".join(f'<text x="{x(epoch):.1f}" y="{height-bottom+25}" text-anchor="middle" class="axis">{epoch}</text>' for epoch in range(20))
    circles = "".join(f'<circle cx="{x(epoch):.1f}" cy="{y(loss):.1f}" r="4" class="point"/>' for epoch, loss in enumerate(LOSSES))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Recorded training loss across 20 epochs</title>
<desc id="desc">Final displayed mini-batch loss decreased from 4.051 at epoch 0 to 2.086 at epoch 19. These are not validation or epoch-average values.</desc>
<style>.title{{font:700 26px Arial,sans-serif;fill:#172b4d}}.axis{{font:13px Arial,sans-serif;fill:#42526e}}.label{{font:14px Arial,sans-serif;fill:#172b4d}}.note{{font:13px Arial,sans-serif;fill:#42526e}}.grid{{stroke:#dfe1e6;stroke-width:1}}.line{{fill:none;stroke:#173f5f;stroke-width:3}}.point{{fill:#173f5f}}.best{{fill:#c0392b}}</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="35" class="title">Recorded Training Loss</text>
{''.join(horizontal)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#42526e"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#42526e"/>
<polyline points="{points}" class="line"/>{circles}
<line x1="{left}" y1="{y(1.25):.1f}" x2="{width-right}" y2="{y(1.25):.1f}" stroke="#6c757d" stroke-dasharray="6 5"/>
<circle cx="{x(19):.1f}" cy="{y(LOSSES[-1]):.1f}" r="7" class="best"/><text x="{x(19)-8:.1f}" y="{y(LOSSES[-1])-14:.1f}" text-anchor="end" class="label">Epoch 19: 2.086</text>
{ticks}<text x="{width/2}" y="{height-40}" text-anchor="middle" class="label">Epoch checkpoint</text><text x="25" y="{height/2}" text-anchor="middle" transform="rotate(-90 25 {height/2})" class="label">Cross-entropy loss</text>
<text x="{left}" y="{height-12}" class="note">Each point is the final displayed mini-batch loss, not an epoch average or validation loss.</text><line x1="{width-310}" y1="{y(1.25):.1f}" x2="{width-280}" y2="{y(1.25):.1f}" stroke="#6c757d" stroke-dasharray="6 5"/><text x="{width-274}" y="{y(1.25)+5:.1f}" class="note">Approx. smoothing floor: 1.25</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")
    print(f"Wrote {output}")

    figure, axis = plt.subplots(figsize=(10.8, 5.6), layout="constrained")
    epochs = list(range(len(LOSSES)))
    axis.plot(epochs, LOSSES, marker="o", color="#173f5f", linewidth=2.5)
    axis.scatter([19], [LOSSES[-1]], color="#c0392b", s=65, zorder=3)
    axis.axhline(1.25, color="#6c757d", linestyle="--", label="Approx. smoothing floor: 1.25")
    axis.annotate("Epoch 19: 2.086", (19, LOSSES[-1]), xytext=(-8, 14), textcoords="offset points", ha="right")
    axis.set(
        title="Recorded Training Loss",
        xlabel="Epoch checkpoint",
        ylabel="Cross-entropy loss",
        xlim=(0, 19),
        ylim=(1.0, 4.5),
    )
    axis.set_xticks(epochs)
    axis.grid(axis="y", color="#dfe1e6")
    axis.legend(loc="lower left")
    figure.text(0.5, 0.01, "Each point is the final displayed mini-batch loss, not an epoch average or validation loss.", ha="center", fontsize=9)
    figure.savefig(png_output, dpi=200)
    plt.close(figure)
    print(f"Wrote {png_output}")


if __name__ == "__main__":
    main()
