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
    {'suffix': '5h1t', 'primary': False},
    {'suffix': 'shat', 'primary': False},
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
                            print(f"🎉 Found match: {ipns_address} to {text_suffix}: with {target_info}")
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
    python ~/code/shitspotter/dev/poc/ipns_vanity.py

    Note: ran for  2 days, 8:34:3[2

    with
        │ partial_match_hist:           │
        │   1: 68889        │
        │   2: 1861    │
        │   3: 55       │
        │ num_keys_checked: 2539916    │
        │ num_cleanups: 2538     │
        │ best_address: k51qzi5uqu5dklsefqn1j2cdfs50fzobbne40r3mtlm0f0h1zccgzk98hvihit     │


    🔍 Found top scoring match: k51qzi5uqu5dgntcdk6a41hsl3rmqsdunqg0ojrfrmjep2ey4lq9psq4s77h1t to 5h1t
    🔍 Found top scoring match: k51qzi5uqu5dlth41fpl038clf7ucfup5tbz8p1g7hfr6nddqcm84up8o0soop to poop

    ⏹️  Stopping...
    ╭──────────────────────────────────────────────────────────────────────────────────────
    │ num_keys_checked: 331042                                                             │
    │ num_cleanups: 65                                                                     │
    │ substates:                                                                           │
    │   shit:                                                                              │
    │     partial_score_hist:                                                              │
    │       1: 35536                                                                       │
    │       2: 1481                                                                        │
    │       3: 32                                                                          │
    │     best_score: 3                                                                    │
    │     best_address: k51qzi5uqu5dga9xtfxz5l3v3u1l9bvvm9absavjkd99ag7cms07hrtg3x1hit     │
    │     best_match: 1hit                                                                 │
    │   scat:                                                                              │
    │     partial_score_hist:                                                              │
    │       1: 35130                                                                       │
    │       2: 1504                                                                        │
    │       3: 26                                                                          │
    │     best_score: 3                                                                    │
    │     best_address: k51qzi5uqu5di05qxz3ss4hlawj7legxrt09amtwof87119uljxxmhxh64slat     │
    │     best_match: slat                                                                 │
    │   poop:                                                                              │
    │     partial_score_hist:                                                              │
    │       1: 34607                                                                       │
    │       2: 1524                                                                        │
    │       3: 25                                                                          │
    │     best_score: 3                                                                    │
    │     best_address: k51qzi5uqu5dlth41fpl038clf7ucfup5tbz8p1g7hfr6nddqcm84up8o0soop     │
    │     best_match: soop                                                                 │
    │   5h1t:                                                                              │
    │     partial_score_hist:                                                              │
    │       1: 35309                                                                       │
    │       2: 1509                                                                        │
    │       3: 23                                                                          │
    │     best_score: 3                                                                    │
    │     best_address: k51qzi5uqu5dgntcdk6a41hsl3rmqsdunqg0ojrfrmjep2ey4lq9psq4s77h1t     │
    │     best_match: 7h1t                                                                 │
    │   shat:                                                                              │
    │     partial_score_hist:                                                              │
    │       1: 35264                                                                       │
    │       2: 1446                                                                        │
    │       3: 28                                                                          │
    │     best_score: 3                                                                    │
    │     best_address: k51qzi5uqu5dks8om1zkgwqndegoksgxx573ykechdqh5vafrooxg6vz57sh1t     │
    │     best_match: sh1t                                                                 │
    │                                                                                      │
    ╰───────────────────────────────────────────────────────────────────────────────────────
    searching ⠧ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 333347/? 13.95 Hz eta  total 7:04:16

    I feel like I'm getting very unlucky.

    Also, running on my other much older machine is giving much much faster
    hash rates. 41Hz vs 13Hz. No idea why this is. Both are on IPFS version
    0.30.0

    Turning off a bunch of other procs

    OH, it must be that IPFS_PATH is pointing to an hdd versus an ssd

    export IPFS_PATH=$HOME/tmp/ipfs-test
    mkdir -p "$IPFS_PATH"
    ipfs init

    And now that makes things go much much faster: 35Hz, but still slower than
    my other machine. Weird. The old machine is using badgerds and the new
    machine is using leveldb, mabye that it is the issue.

    The new machine also started to slow down to 20 hz

    # But maybe we can run a few in parallel now
    export IPFS_PATH=$HOME/tmp/ipfs-test2
    mkdir -p "$IPFS_PATH"
    ipfs init
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    # But maybe we can run a few in parallel now
    export IPFS_PATH=$HOME/tmp/ipfs-test3
    mkdir -p "$IPFS_PATH"
    ipfs init
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    # But maybe we can run a few in parallel now
    export IPFS_PATH=$HOME/tmp/ipfs-test4
    mkdir -p "$IPFS_PATH"
    ipfs init
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    Hmm, maybe not, that caused a significant drop in hash rate, down to 10Hz,
    but maybe that is ok if there are enough workers?

    export IPFS_PATH=$HOME/tmp/ipfs-test5
    mkdir -p "$IPFS_PATH"
    ipfs init --profile badgerds
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    Oh yeah, using badgerds is much quicker, and it looks like it can run in
    parallel without too much issue.

    export IPFS_PATH=$HOME/tmp/ipfs-test6
    mkdir -p "$IPFS_PATH"
    ipfs init --profile badgerds
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    export IPFS_PATH=$HOME/tmp/ipfs-test7
    mkdir -p "$IPFS_PATH"
    ipfs init --profile badgerds
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    export IPFS_PATH=$HOME/tmp/ipfs-test8
    mkdir -p "$IPFS_PATH"
    ipfs init --profile badgerds
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py

    export IPFS_PATH=$HOME/tmp/ipfs-test9
    mkdir -p "$IPFS_PATH"
    ipfs init --profile badgerds
    python ~/code/shitspotter/dev/poc/ipns_vanity_v2.py


    """
    main()
