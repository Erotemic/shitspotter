import subprocess
import threading
import queue
import random
import string
import Levenshtein
import ubelt as ub
from os.path import commonprefix

# Configurations
DELETE_THRESHOLD = 5000  # When to pause and delete keys
NUM_GENERATORS = 1  # Number of parallel key generators

key_queue = queue.Queue(maxsize=100000)  # Limit queue size to avoid memory buildup
stop_event = threading.Event()

# Note: wont work if we can actually start to call key generator multiple
# times, but I think that's not allowed anyway. Is event what we want?
delete_lock = threading.Lock()  # Separate lock for deletion
semaphore = threading.Semaphore(1)  # Initial value of 1 (unlocked)


# List of targets vanity suffixes that would be nice.
vanity_target_infos = [
    {'suffix': 'shit', 'primary': True},
    {'suffix': 'scat', 'primary': False},
    {'suffix': 'poop', 'primary': False},
    {'suffix': '5h1t', 'primary': True},
    {'suffix': 'shat', 'primary': True},
]


def generate_key():
    """Generate an IPFS key and return its name and IPNS address."""
    key_name = "vanity-key-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
    result = subprocess.run(["ipfs", "key", "gen", "--type=ed25519", key_name],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(repr(result))
    address = result.stdout.strip()
    return key_name, address


def key_generator():
    """Continuously generates keys and adds them to the queue."""
    while not stop_event.is_set():
        if delete_lock.locked():
            continue  # Wait for deletion to finish
        with semaphore:
            try:
                result = generate_key()
            except Exception:
                continue
            else:
                key_queue.put(result)


def common_suffix(s1, s2):
    """Compute the common suffix of two strings."""
    return ''.join(reversed(commonprefix([s1[::-1], s2[::-1]])))


def check_suffix_match_quality(text, target_suffix):
    """
    Ignore:
        text = 'k51qzi5uqu5dh9ps31f0hzub6shd1jhcjjrvzc1kzj1u196fb5eb51kkndvhit'
        target = 'shit'
        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('time'):
            with timer:
                suffix = text[-len(target):]
                score = Levenshtein.distance(suffix, target)
        for timer in ti.reset('time'):
            with timer:
                common_suffix(suffix, target)
    """

    text_suffix = text[-len(target_suffix):]
    distance = Levenshtein.distance(text_suffix, target_suffix)
    score = len(target_suffix) - distance
    return score, text_suffix


def main():
    print(f"🚀 Searching for IPNS vanity suffix '{vanity_target_infos}'...")

    # Start multiple key generator threads
    generators = [threading.Thread(target=key_generator, daemon=True) for _ in range(NUM_GENERATORS)]
    print(f'generators={generators}')
    for thread in generators:
        thread.start()

    prog = ub.ProgIter(desc='Searching')
    prog.begin()

    delete_list = []

    state = {
        'num_keys_checked': 0,
        'num_cleanups': 0,
    }
    substates = {}
    state['substates'] = substates
    for target_info in vanity_target_infos:
        suffix = target_info['suffix']
        substates[suffix] =  {
            'partial_score_hist': ub.ddict(lambda: 0),
            'best_score': 0,
            'best_address': None,
            'best_key': None,
        }

    import kwutil
    pman = kwutil.ProgressManager()
    with pman:
        prog = pman.ProgIter(desc='searching')
        prog.begin()

        try:
            while not stop_event.is_set():
                key_name, ipns_address = key_queue.get()

                semi_relevant = False

                for target_info in vanity_target_infos:
                    target_suffix = target_info['suffix']

                    substate = state['substates'][target_suffix]
                    best_score = substate['best_score']
                    score, text_suffix = check_suffix_match_quality(ipns_address, target_suffix)

                    # match = common_suffix(ipns_address, VANITY_SUFFIX)
                    if score > 1:
                        semi_relevant = True

                    if score > 0:
                        substate['partial_score_hist'][score] += 1

                    if score >= best_score:
                        prog.ensure_newline()
                        substate['best_score'] = score
                        substate['best_match'] = text_suffix
                        substate['best_address'] = ipns_address
                        substate['best_key'] = key_name
                        prog.update_info(kwutil.Yaml.dumps(state))
                        if text_suffix == target_suffix and target_info['primary']:
                            print(f"🎉 Found match: {ipns_address}")
                            subprocess.run(["ipfs", "key", "export", "--output", f"{key_name}.pem", key_name])
                            stop_event.set()
                            break
                        else:
                            print(f"🔍 Found top scoring match: {ipns_address} to {target_suffix}")

                if not semi_relevant:
                    delete_list.append(key_name)

                # Pause generation, delete old keys, then resume
                if len(delete_list) >= DELETE_THRESHOLD:
                    state['num_cleanups'] += 1
                    prog.update_info(kwutil.Yaml.dumps(state))
                    prog.ensure_newline()
                    with delete_lock:
                        with semaphore:
                            delete_keys(delete_list)

                state['num_keys_checked'] += 1
                prog.step()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
            stop_event.set()

        prog.end()

    # Final cleanup
    print('Final cleanup')
    with delete_lock:
        with semaphore:
            delete_keys(delete_list)
    prog.end()


def delete_keys(delete_list):
    # print('Intermediate cleanup')
    # print(f"🔍 Best match so far: {best_match}")
    # print(f'🧹 Cleanup {len(delete_list)} keys')
    for chunk in ub.chunks(delete_list, chunksize=100):
        out = ub.cmd(["ipfs", "key", "rm"] + chunk)
        if out.returncode != 0:
            raise Exception(repr(out))
    delete_list.clear()


def external_cleanup():
    """
    Cleanup extra keys outside this program if something goes bad
    """
    import ubelt as ub
    result = ub.cmd('ipfs key list -l', verbose=3)
    to_delete = []
    to_keep = []
    for line in result.stdout.strip().split('\n'):
        ipns_address, key = line.strip().split(' ')
        if key.startswith('vanity'):
            match = common_suffix(ipns_address, VANITY_SUFFIX)
            if len(match) < 2:
                to_delete.append(key)
            else:
                to_keep.append((ipns_address, key))
    delete_keys(to_delete)

if __name__ == "__main__":
    """
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    Note: ran for  2 days, 8:34:3[2

    with
        │ partial_match_hist:           │
        │   1: 68889        │
        │   2: 1861    │
        │   3: 55       │
        │ num_keys_checked: 2539916    │
        │ num_cleanups: 2538     │
        │ best_address: k51qzi5uqu5dklsefqn1j2cdfs50fzobbne40r3mtlm0f0h1zccgzk98hvihit     │
        │ best_key: vanity-key-rd45u2c5f2i5s4sfxes9nhcs5rne1yiq │
    """
    main()
