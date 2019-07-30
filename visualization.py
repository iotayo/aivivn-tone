import visdom


class Visualization:
    def __init__(self, env_name, xlabel, ylabel, title):
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def plot_line(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                title=self.title,
            )
        )
