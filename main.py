import click


@click.group()
def _trainer():
    """ Trainner groups"""
    pass

@click.group()
def _new():
    """ Create new model """


_trainer.