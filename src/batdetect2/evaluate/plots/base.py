from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from batdetect2.core import BaseConfig
from batdetect2.typing import TargetProtocol


class BasePlotConfig(BaseConfig):
    label: str = "plot"
    theme: str = "default"
    title: Optional[str] = None
    figsize: tuple[int, int] = (5, 5)
    dpi: int = 100


class BasePlot:
    def __init__(
        self,
        targets: TargetProtocol,
        label: str = "plot",
        figsize: tuple[int, int] = (5, 5),
        title: Optional[str] = None,
        dpi: int = 100,
        theme: str = "default",
    ):
        self.targets = targets
        self.label = label
        self.figsize = figsize
        self.dpi = dpi
        self.theme = theme
        self.title = title

    def get_figure(self) -> Figure:
        plt.style.use(self.theme)
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        if self.title is not None:
            fig.suptitle(self.title)

        return fig

    @classmethod
    def build(cls, config: BasePlotConfig, targets: TargetProtocol, **kwargs):
        return cls(
            targets=targets,
            figsize=config.figsize,
            dpi=config.dpi,
            theme=config.theme,
            label=config.label,
            title=config.title,
            **kwargs,
        )
