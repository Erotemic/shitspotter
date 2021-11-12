

def main():
    """
    """
    import os
    import dateutil.parser
    from os.path import join
    dpath = '/data/store/data/shit-pics/'

    total_items = 0
    num_images = 0

    gpath_list = []

    for r, ds, fs in os.walk(dpath):
        to_remove = []
        for idx, d in enumerate(ds):
            if not d.startswith('poop-'):
                to_remove.append(idx)

        for idx in to_remove[::-1]:
            del ds[idx]

        change_point = dateutil.parser.parse('2021-05-11T120000')

        dname = os.path.relpath(r, dpath)
        if dname.startswith('poop-'):
            timestr = dname.split('poop-')[1]
            timestamp = dateutil.parser.parse(timestr)

            is_double = timestamp < change_point
            is_triple = not is_double

            for fname in fs:
                gpath = join(r, fname)
                gpath_list.append(gpath)

            num_files = len(fs)
            num_images += num_files
            if is_triple:
                num_items = num_files // 3
            else:
                num_items = num_files // 2
            print('num_items = {!r}'.format(num_items))
            total_items += num_items

    print('num_images = {!r}'.format(num_images))
    print('total_items = {!r}'.format(total_items))

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/scripts/gather_shit.py
        python /data/store/data/shit-pics/gather_shit.py
    """
    main()
