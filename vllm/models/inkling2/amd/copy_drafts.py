"""Use an unambiguous exact continuation as a native greedy DSpark draft."""

from vllm.triton_utils import tl, triton


@triton.jit
def _replace_copy_drafts(
    History,
    Length,
    Sampled,
    Temperature,
    Drafts,
    SCAN: tl.constexpr,
    HISTORY_SIZE: tl.constexpr,
    SUFFIX: tl.constexpr,
    DRAFTS: tl.constexpr,
):
    length = tl.load(Length)
    if (tl.load(Sampled) <= 0) | (tl.load(Temperature) != 0):
        return
    length = tl.minimum(length, HISTORY_SIZE)
    if length < 2 * SUFFIX + DRAFTS:
        return

    suffix_start = length - SUFFIX
    start = tl.arange(0, SCAN)
    # Previous matches and their complete continuations must end before
    # the current suffix. The history contains only committed token IDs.
    match = start + SUFFIX + DRAFTS <= suffix_start
    for i in tl.static_range(SUFFIX - 1, -1, -1):
        token = tl.load(History + start + i, match, other=-1)
        suffix = tl.load(History + suffix_start + i)
        match = match & (token == suffix)

    last_match = tl.max(tl.where(match, start, -1), axis=0)
    if last_match < 0:
        return

    # Disagreement anywhere in the seven-token continuation leaves all
    # native proposals intact. No probabilistic draft scores are changed.
    different = tl.full((SCAN,), False, tl.int1)
    for i in tl.static_range(DRAFTS):
        token = tl.load(History + start + SUFFIX + i, match, other=-1)
        expected = tl.load(History + last_match + SUFFIX + i)
        different |= match & (token != expected)
    if tl.sum(different.to(tl.int32), axis=0) == 0:
        slots = tl.arange(0, 8)
        tokens = tl.load(History + last_match + SUFFIX + slots, slots < DRAFTS, other=0)
        tl.store(Drafts + slots, tokens, slots < DRAFTS)


def replace_copy_drafts(history, length, sampled, temperature, drafts):
    """Replace one contiguous seven-token row in place when the suffix matches."""
    _replace_copy_drafts[(1,)](
        history, length, sampled, temperature, drafts,
        triton.next_power_of_2(history.shape[1]), history.shape[1], 12, drafts.shape[1], num_warps=8,
    )
