"""Three illustrative demo patients (made up, not real people)."""

_HEALTHY = dict(age=45, sex="M", weight=80, bp=70, sg=1.020, al=0, su=0, bgr=100, bu=20, sc=0.9, sod=140,
                pot=4.2, hemo=15.0, pcv=45, wc=7500, rc=5.2, rbc=0, pc=0, pcc=0, ba=0, htn=0, dm=0,
                cad=0, appet=0, pe=0, ane=0)

CASES = {
    "Healthy adult (eGFR >= 90)": _HEALTHY,
    "CKD stage G3b": dict(_HEALTHY, age=68, weight=78, bp=90, sg=1.015, al=2, su=1, bgr=140, bu=48, sc=1.9,
                          sod=138, pot=4.8, hemo=11.2, pcv=34, wc=8500, rc=4.2, rbc=1, pc=1, htn=1, dm=1),
    "CKD stage G5": dict(_HEALTHY, age=72, sex="F", weight=62, bp=100, sg=1.010, al=4, bgr=120, bu=110, sc=3.6,
                         sod=134, pot=5.6, hemo=8.4, pcv=26, wc=9800, rc=3.1, rbc=1, pc=1, pcc=1, htn=1,
                         appet=1, pe=1, ane=1),
}
