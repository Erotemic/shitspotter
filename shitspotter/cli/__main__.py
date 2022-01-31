
def analysis():
    """
    CommandLine:
        python -m shitspotter analysis
    """
    from shitspotter.plots import update_analysis_plots
    update_analysis_plots()


def main():
    import shitspotter
    print('shitspotter.__version__ = {!r}'.format(shitspotter.__version__))
    print('shitspotter.__file__ = {!r}'.format(shitspotter.__file__))
    import sys
    sys.argv
    if 'analysis' in sys.argv:
        analysis()

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/__main__.py
    """
    main()
