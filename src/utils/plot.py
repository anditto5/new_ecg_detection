# visualization utilities (curves, prediction results)
import matplotlib.pyplot as plt

def plot_dummy_curve():
    """Example plot function (replace with real training curves)"""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9], label="Dummy Curve")
    ax.set_title("Dummy Training Curve")
    ax.legend()
    return fig
