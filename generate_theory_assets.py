from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report_assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc, ec="#1f2937"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="#374151",
        )
    )


def eta_chain():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("Commande\ne-commerce", "#e0f2fe"),
        ("Promesse\nETA", "#dbeafe"),
        ("Préparation\net tri", "#ecfccb"),
        ("Transport\ninterurbain", "#fef3c7"),
        ("Dernier\nkilomètre", "#ffedd5"),
        ("Réception\nclient", "#fee2e2"),
    ]
    xs = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    for i, ((label, color), x) in enumerate(zip(steps, xs)):
        add_box(ax, x, 0.55, 0.13, 0.22, label, color)
        if i < len(steps) - 1:
            arrow(ax, x + 0.13, 0.66, xs[i + 1], 0.66)

    risks = [
        ("Volatilité\nde la demande", 0.10),
        ("Capacité\ndu transporteur", 0.43),
        ("Congestion,\nmétéo, client", 0.74),
    ]
    for text, x in risks:
        add_box(ax, x, 0.16, 0.18, 0.16, text, "#f8fafc", "#64748b")
        arrow(ax, x + 0.09, 0.32, x + 0.09, 0.55)

    ax.text(
        0.5,
        0.93,
        "Chaîne de formation et de dégradation de la promesse ETA",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "21_theorie_chaine_eta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ml_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    top = [
        ("Données\nbrutes", "#e0f2fe"),
        ("Préparation\nanti-fuite", "#dbeafe"),
        ("Encodage\npipeline", "#ecfccb"),
        ("Modèles\ncandidats", "#fef3c7"),
        ("Validation\ntemporelle", "#ffedd5"),
        ("Déploiement\nStreamlit", "#fee2e2"),
    ]
    xs = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    for i, ((label, color), x) in enumerate(zip(top, xs)):
        add_box(ax, x, 0.58, 0.13, 0.20, label, color)
        if i < len(top) - 1:
            arrow(ax, x + 0.13, 0.68, xs[i + 1], 0.68)

    add_box(ax, 0.22, 0.18, 0.22, 0.16, "Baseline métier\nETA existante", "#f8fafc", "#64748b")
    add_box(ax, 0.56, 0.18, 0.22, 0.16, "Interprétation\nimportance + limites", "#f8fafc", "#64748b")
    arrow(ax, 0.33, 0.34, 0.57, 0.58)
    arrow(ax, 0.67, 0.34, 0.74, 0.58)
    ax.text(
        0.5,
        0.93,
        "Architecture méthodologique du pipeline prédictif",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "22_theorie_pipeline_ml.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    eta_chain()
    ml_pipeline()
    print("theory_assets_ok")
