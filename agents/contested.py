'''
contested.py - triage BEFORE the judge . the contested decision function
decides how much EFFORT a fight deserves , never who wins it .

the three routes (designed , mostly deferred) :
- "ignore"      : park the pair , stay contested , the report shows both
                  sides . for fights whose resolution changes nothing .
- "collapse"    : send the pair to the adjudicator for resolution now .
- "investigate" : spawn a contradiction-triggered verify task first - the
                  ONLY legal birthplace of kind="verify" , depth cap 1 .

v1 STUB : every pair routes to "collapse" . the ignore/investigate
thresholds are DEFERRED on purpose - no run data yet justifies any
threshold value (generalize on the second instance , defer from evidence
not imagination) . the disabled placeholder lives in parametres.py as
INVESTIGATE_FANOUT_THRESHOLD = None ; when real fan-out data from runs
exists , the triage below grows its two other branches .

detector vs judge vs triage : the evaluator DETECTS pairs , this file
sizes the EFFORT , the adjudicator JUDGES . none of the three ever does
another's job .
'''

from parametres import INVESTIGATE_FANOUT_THRESHOLD


VALID_TRIAGE_ROUTES = ("ignore", "collapse", "investigate")


def triage_contested_pairs(workspace, contested_pairs):
    # pure code , no llm . returns every pair sorted into its route .
    routed = {
        "ignore": [],
        "collapse": [],
        "investigate": [],
    }

    for pair in contested_pairs:
        # v1 : every fight gets resolved now . the seam for the real CDF :
        # compute fan-out (dependents per claim , derived from provenance)
        # and compare against INVESTIGATE_FANOUT_THRESHOLD - which stays
        # None (disabled) until run data justifies a value
        if INVESTIGATE_FANOUT_THRESHOLD is None:
            routed["collapse"].append(pair)
        else:
            # unreachable while the threshold is disabled - kept explicit
            # so enabling the constant forces this branch to be written
            raise NotImplementedError(
                "investigate-first triage is designed but not built - "
                "disable INVESTIGATE_FANOUT_THRESHOLD or implement the branch"
            )

    return routed
