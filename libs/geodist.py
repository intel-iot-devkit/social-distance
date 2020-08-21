"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from shapely.geometry import LineString, Point, Polygon
import math

def social_distance(frame_shape, a, b, c, d, min_iter=3, min_w=None, max_w=None):

    h, w = frame_shape
    A, B ,C ,D = Point(a), Point(b), Point(c), Point(d)
    AB = get_line(A, B)
    CD = get_line(C, D)

    COEF = 1

    minx = A.x if A.x < C.x else C.x
    maxx = B.x if B.x > D.x else D.x
    if min_w * 1.8 <= max_w and minx <= w * .1:
        return {"euclidean": True, "alert": False, "distance": 0}

    in_border = True if minx < w * .3 or maxx > w - (w * .3) else False
    thr = .1 if in_border else .01
    if abs(CD.length - AB.length) <= thr or abs(C.y - A.y) <= h*.01:
        p = ((CD.length + AB.length / 2) - min_w) / max_w
        if p < .3 :
            COEF = 1.0 + (1 - p)

        result = euclidean_distance(AB, CD, min_iter, COEF)
        return result

    # Calculo pendiente ordenada al origen de la recta BD
    bd_a, bd_k = get_line_component(B, D)
    ac_a, ac_k = get_line_component(A, C)

    bdinf = -9999999999 if B.x < D.x else 9999999999
    BDinf = get_line(D, Point(bdinf, get_y(bdinf, bd_a, bd_k)))

    acinf = -9999999999 if A.x < C.x else 9999999999
    ACinf = get_line(C, Point(acinf, get_y(acinf, ac_a, ac_k)))
    # PUNTO DE FUGA
    PF = BDinf.intersection(ACinf)
    euclidean = False
    try:
        inter = list(PF.coords)
    except Exception as e:
        euclidean =True

    if euclidean or not list(PF.coords):
        p = ((CD.length + AB.length / 2) - min_w) / max_w
        if p < .3:
            COEF = 1.0 + (1 - p)

        result = euclidean_distance(AB, CD, min_iter, COEF)
        return result

    E = Point(get_x(h, ac_a, ac_k), h)
    F = Point(get_x(h, bd_a, bd_k), h)

    init_iter = 1
    if E.y - C.y < 1:
        if bdinf > 0:
            EPF = get_line(E, PF)
            new_c = cut(EPF, CD.length)[0]
            new_c = list(new_c.coords)
            new_c = new_c[1]
            if A.y < new_c[1]:
                init_iter += 1
                C = Point(new_c[0], new_c[1])

            else:
                return {"euclidean": False, "alert": False, "iterations": init_iter}

        else:
            FPF = get_line(F, PF)
            new_d = cut(FPF, CD.length)[0]
            new_d = list(new_d.coords)
            new_d = new_d[1]
            if B.y < new_d[1]:
                init_iter += 1
                D = Point(new_d[0], new_d[1])

            else:
                return {"euclidean": False, "alert": False, "iterations": init_iter}

    if bdinf > 0:
        try:
            Z = get_line(F, PF)
        except Exception:
            print(list(F.coords))
            print(list(PF.coords))
            raise
        frac = Z.length / 2
        l1, l2 = cut(Z, frac)
        if l2.contains(B):
            med = Point(list(l2.coords)[0])
            med_b = get_line(med, B).length
            med_pf = get_line(med, PF).length
            dist = med_b / med_pf
            COEF = math.exp(1 + dist)

    else:
        Z = get_line(E, PF)
        frac = Z.length / 2

        l1, l2 = cut(Z, frac)
        if l2.contains(A):
            med = Point(list(l2.coords)[0])
            med_a = get_line(med, A).length
            med_pf = get_line(med, PF).length
            dist = med_a / med_pf
            COEF = math.exp(1 + dist)



    cnt = get_distance(PF, E, F, C, D, A, B, bdinf, init_iter, COEF)
    alert = False if cnt >= min_iter else True
    result = {"euclidean": False, "alert": alert, "iterations": cnt}
    return result


def get_line_component(p1,p2):
    run = (p2.x - p1.x) * 1.0
    rise = (p2.y - p1.y) * 1.0
    try:
        a = rise/run
    except ZeroDivisionError:
        a = rise/0.0000001

    # ordenada al origen k
    # k = y - (a * x)
    k = p1.y - (a * p1.x)
    return a, k


def get_line(A, B):
    return LineString([A,B])


def get_x(y , a, k):
    return (y - k) / a


def get_y(x, a, k):
    return a * x + k


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def get_distance(PF, E, F, Za, Zb, Zlimit_a, Zlimit_b, inf1_dir, init_iter=1, coef=1):
    _inf = inf1_dir
    inf = inf1_dir * -1
    Za_over_Zlimit = False
    PFE = get_line(PF, E)
    PFF = get_line(PF, F)
    EF = get_line(E, F)
    len_EF = EF.length
    cnt = init_iter

    while not Za_over_Zlimit:
        if inf > 0:
            # inf = 9999
            # _inf = -9999

            len_ZbF = get_line(Zb, F).length
            F_proj = Point(F.x, F.y + len_ZbF)
            E_proj = Point(E.x, F.y + len_ZbF + len_EF)

            proj_a, proj_k = get_line_component(E_proj, F_proj)
            AUX = Point(get_x(F.y, proj_a, proj_k), F.y)

            aux_a, aux_k = get_line_component(Zb, AUX)
            AUXinf = get_line(Point(_inf, get_y(_inf,aux_a, aux_k)), AUX)
            Zaux_a = PFE.intersection(AUXinf)

            if Zaux_a.y < Zlimit_a.y:
                Za_over_Zlimit = True
            else:
                cnt +=1
                Za = Zaux_a
                Zb_aux = get_line(Za ,Point(inf, Za.y))
                Zb = PFF.intersection(Zb_aux)
        else:
            # inf = - 9999
            # _inf = 9999
            len_ZaE = get_line(Za, E).length
            F_proj = Point(F.x, F.y + len_ZaE + len_EF)
            E_proj = Point(E.x, F.y + len_ZaE)

            proj_a, proj_k = get_line_component(E_proj, F_proj)
            AUX = Point(get_x(E.y, proj_a, proj_k), E.y)

            aux_a, aux_k = get_line_component(Za, AUX)
            AUXinf = get_line(Point(_inf, get_y(_inf, aux_a, aux_k)), AUX)
            Zaux_b = PFF.intersection(AUXinf)

            if Zaux_b.y < Zlimit_b.y:
                Za_over_Zlimit = True
            else:
                cnt +=1
                Zb = Zaux_b
                Za_aux = get_line(Zb ,Point(inf, Zb.y))
                Za = PFE.intersection(Za_aux)
    cnt = cnt * coef
    return cnt


def get_crop(a, b):
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b

    cxmin = axmin if axmin < bxmin else bxmin
    cymin = aymin if aymin < bymin else bymin
    cxmax = axmax if axmax > bxmax else bxmax
    cymax = aymax if aymax > bymax else bymax

    return cxmin, cymin, cxmax, cymax


def euclidean_distance(AB, CD, min_iter, coef):
    Z1 = AB.centroid
    Z2 = CD.centroid
    min_dist = (AB.length * min_iter) * .8
    distance = get_line(Z1, Z2).length * coef
    if distance < AB.length:
        return {"euclidean": True, "alert": False, "distance": distance}

    alert = True if distance < min_dist else False
    return {"euclidean": True, "alert": alert, "distance": distance}


