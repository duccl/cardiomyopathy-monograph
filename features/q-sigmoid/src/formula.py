def q_sigmoid(I,beta=0.3,alfa=0.1,lamb=0.5,q=2):
    if q < 1:
	    tsallis = 1/(1-q)
	    local_intensity = abs(I-beta)/alfa
	    base = (1.0+(lamb*(1-q)*(local_intensity)))
	    gaussian = pow(base,tsallis)
	    resp = 2.0/(1.0+gaussian)
	    return resp
    if I == beta:
	    return 1
    tsallis = 1/(1-q)
    local_intensity = -1*(1/(abs(I-beta)/alfa))
    base = (1.0+(lamb*(1-q)*(local_intensity)))
    gaussian = pow(base,tsallis)
    resp = 1.0/(1.0+gaussian)
    return resp