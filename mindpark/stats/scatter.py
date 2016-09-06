from mindpark.stats.plot import Plot


class Scatter(Plot):

    SCATTER = dict(alpha=0.1, lw=0)

    def _plot(self, ax, domain, line, label, color, marker):
        ax.scatter(domain, line, c=color, marker=marker, **self.SCATTER)

        legend = self.SCATTER.copy()
        legend['alpha'] = 1.0
        ax.scatter([], [], c=color, marker=marker, label=label, **legend)
