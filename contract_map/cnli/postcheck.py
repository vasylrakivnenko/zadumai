"""Final consistency checks on an answer, after the sources agree (2026-10-03, after the sealed test; measured by CV on the
484 tuning NDAs only). Scope and precedence at document level: an override elsewhere cancels a "yes"; a "yes" needs the
question's own concept somewhere in the NDA."""
import re
I = re.I
OVERRIDE_USE = (r"\bfree to use\b|\bresiduals?\b|\bretained in the (?:unaided )?(?:memor\w*|minds?)\b|\bunaided memor|\bshall not be (?:restricted|limited|precluded)\w*\b.{0,80}\buse|"
                r"\bnothing\b.{0,160}\b(?:limit|restrict|prevent|preclude|prohibit)\w*\b.{0,100}\b(?:use|utiliz)|"
                r"\bnot (?:be )?(?:restricted|limited|prevented|precluded) (?:in|from) .{0,40}\buse")
NOTICE = (r"(?:notif|notice|inform|advise)\w*\b.{0,250}\b(?:required|compelled|subpoena|court|law|order|legal)|"
          r"(?:required|compelled|subpoena|court order|by law|legal process|governmental)\b.{0,250}\b(?:notif|notice|inform|advise)")


def keep(q, answer, text):
    if q == "nda-4" and answer == "yes" and re.search(OVERRIDE_USE, text, I): return False
    if q == "nda-8" and answer == "yes" and not re.search(NOTICE, text, I | re.S): return False
    return True
