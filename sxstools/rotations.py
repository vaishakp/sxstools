# --- rotations.py

"""
A collection of functions related to rotating waveforms with quaternions.
This is a part of gwtools that has been copied to SurrogateModeling.
"""

__copyright__ = "Copyright (C) 2015 Jonathan Blackman"
__status__    = "testing"
__author__    = "Jonathan Blackman"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# TODO:
#   1. minRotQuat_to_phases and phases_to_minRotQuat currently use finite
#       differencing for simplicity. The error that is made is that the
#       z-component of dq is not exactly zero, and it is closer to zero
#       if we use a better derivative.
#       We can use splines and take
#           dq = s*dvdt - dsdt*v - cross(v.T, dvdt.T).T
#       where s = q[0], v = q[1:], but then I think we need to do a non-linear
#       integration in order to recover q. Specifically, this dq is
#           lim_{dt->0} {(q(t)^{-1} \times q(t + dt) - 1)/dt}
#       so it's not as simple as just integrating dq.
#       Using finite differencing makes this step simple, we just multiply
#       the small dq with the previous q(t) to obtain q(t+dt), which is why
#       it is currently being done.
#   2. filteredCoprQuat, which makes use of filterQuat, is decent but not
#       perfect: If we compute (quat)*(filteredCoprQuat)^{-1}, we get the
#       rotation that was filtered out.  Ideally, this would be purely
#       nutation effects and we could treat the (x, y) components like an
#       amplitude and phase.  This currently doesn't happen - the filtering
#       isn't so good near extrema of the quaternion components, and the
#       filtered component goes outside the range of oscillation. A more
#       sophisticated method of filtering might let us model the nutation
#       of the coprecessing frame.

import warnings as _warnings

import numpy as np

import scri
import quaternion

from scipy.interpolate import UnivariateSpline as _spline

# In some of the ringdowns we find that coprecessing frame quaternions jump
# around wildly as the amplitude becomes small and it becomes hard to define a
# coprecessing frame. So, we taper off the quaternions to constant values in
# coprecessingQuatAndWaveform(t, h). These set the default values for the star
# and end of the tapering, where t=0 is assumed to be peak amplitude. These
# should be fine for NR runs, but you probably don't want any tapering for PN.
COPR_TAPER_START_TIME_DEFAULT = 0
COPR_TAPER_END_TIME_DEFAULT = 20
COPR_TAPER_TRANSITION_TIMES_DEFAULT = [COPR_TAPER_START_TIME_DEFAULT,
                                       COPR_TAPER_END_TIME_DEFAULT]

# num_modes given lMax, given by (lMax -1) * (lMax + 3)
# To get the number of modes for a given lmax, do nModesVsLMax[lmax],
# for some lmax >= 2.
nModesVsLMax = [-3, 0, 5, 12, 21, 32, 45, 60, 77]

#------------------------------------------------------------------------
def rotate_xy(v, phase):
    """Rotate vector v around by phase radians about the z-axis.
    """
    res = 1. * v
    cp = np.cos(phase)
    sp = np.sin(phase)
    res[0] = v[0]*cp + v[1]*sp
    res[1] = -v[0]*sp + v[1]*cp
    return res

#------------------------------------------------------------------------
def rotate_spin(chi_copr, orb_phase):
    """ Rotate spin vector from coprecessing frame to coorbital frame.
    """
    v = chi_copr.T
    sp = np.sin(orb_phase)
    cp = np.cos(orb_phase)
    res = 1. * v
    res[0] = v[0] * cp + v[1] * sp
    res[1] = v[1] * cp - v[0] * sp
    chi_coorb = res.T
    return chi_coorb

#------------------------------------------------------------------------
def multiplyQuats(q1, q2):
    """q1, q2 must be [scalar, x, y, z] but those may be arrays or scalars"""
    return np.array([
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[2]*q2[3] - q2[2]*q1[3] + q1[0]*q2[1] + q2[0]*q1[1],
            q1[3]*q2[1] - q2[3]*q1[1] + q1[0]*q2[2] + q2[0]*q1[2],
            q1[1]*q2[2] - q2[1]*q1[2] + q1[0]*q2[3] + q2[0]*q1[3]])

#------------------------------------------------------------------------
def quatInv(q):
    """Returns QBar such that Q*QBar = 1"""
    qConj = -q
    qConj[0] = -qConj[0]
    normSqr = multiplyQuats(q, qConj)[0]
    return qConj/normSqr

#-------------------------------------------------------------------------
def minRotQuat_to_phases(quat, iAlign=0):
    """Given quaternions vs. time for a minimally rotating frame (no
    instantaneous rotation in the z-direction),
    returns (quatPhase, dirPhase)
    where quatPhase is the overall precession phase and dirPhase gives the
    instantaneous precession direction in the (x, y) plane.
    Note that len(dirPhase) == len(quat[0]) - 1
    """
    identity_err = np.sqrt(sum(abs(quat[:,iAlign] - np.array([1., 0., 0., 0.]))**2.))
    if identity_err > 1.e-8:
        if identity_err > 1.e-3:
            raise Exception('Init quat differs from identity by %s'%identity_err)
        _warnings.warn('Init quat differs slightly from identity')

    # Work with finite differences and cumsums throughout
    dq = multiplyQuats(quatInv(quat[:,:-1]), quat[:,1:])[1:]
    omegaDt = np.sqrt(np.sum(dq**2, 0))
    quatPhase = np.append(0., np.cumsum(omegaDt))
    eps = 1.e-12*max(abs(omegaDt))
    omegaNoZeros = 0.5*(
            omegaDt + 1.e-10*np.ones(len(omegaDt)) +
            abs(omegaDt - 1.e-10*np.ones(len(omegaDt))) )
    vecs = dq/omegaNoZeros
    if max(abs(vecs[2])) > 1.e-4:
        _warnings.warn('Got a z-component of %s! Is this a min. rot. quat?'%(
                max(abs(vecs[2]))))
    dirPhase = np.unwrap(np.angle(vecs[0] + 1.j*vecs[1]))
    return quatPhase, dirPhase

#-------------------------------------------------------------------------
def phases_to_minRotQuat(quatPhase, dirPhase, iAlign=0):
    """Inverts minRotQuat_to_phases"""
    omega = np.diff(quatPhase)
    dq = omega*np.array([
            np.zeros(len(dirPhase)),
            np.cos(dirPhase),
            np.sin(dirPhase),
            np.zeros(len(dirPhase))])
    dq[0] = np.sqrt(np.ones(len(dirPhase)) - omega**2.)
    newQuats = [np.array([1., 0., 0., 0.])]
    for q in dq.T:
        newQuats.append(multiplyQuats(newQuats[-1], q))
    newQuats = np.array(newQuats).T
    if iAlign != 0:
        q0 = newQuats[:,iAlign]
        newQuats = multiplyQuats(quatInv(q0), newQuats)
    return newQuats

#-------------------------------------------------------------------------
def lMax_from_num_modes(num_modes):
    """Assuming h is provided as an array of modes, with num_modes = len(h),
    and the modes are sorted as
    [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2), ...],
    we take num_modes as input and return the lMax of the mode array h.
    """
    # Fancy way to invert the relation. Jon comes up with these stuff, don't
    # ask me.
    lMax = nModesVsLMax.index(num_modes)
    return lMax


#------------------------------------------------------------------------
def coprecessingQuat(t, h, iAlign=None,
        transition_times=COPR_TAPER_TRANSITION_TIMES_DEFAULT):
    """Uses scri to find the coprecessing frame of h
    Specify iAlign if the quat should be aligned about the z-axis"""

    quat, wave = coprecessingQuatAndWaveform(t, h,
            transition_times=transition_times)

    if iAlign is not None:
        q0 = quat[:,iAlign]
        if abs(q0[1]) + abs(q0[2]) > 1.e-8:
            raise Exception("Expected quaternion at iAlign to be a " +
                            "z-rotation, got %s"%q0)
        qz = zRotationQuat(-2*np.arctan2(q0[3], q0[0]))
        quat = multiplyQuats(qz, quat) # Non-commutative! Is this the better way?
    return quat

def coprecessingQuatAndWaveform(t, h,
        transition_times=COPR_TAPER_TRANSITION_TIMES_DEFAULT,
        use_news=False):
    """ Transfroms from inertial frame to coprecessing frame using scri.
    """

    w = scri.WaveformModes(
          dataType = scri.h,
          t = np.copy(t),
          data = np.copy(h.T),      # Because data gets modified internally
          ell_min = 2,
          ell_max = lMax_from_num_modes(len(h)),
          frameType = scri.Inertial,
          r_is_scaled_out = True,
          m_is_scaled_out = True,
    )

    if not use_news:
        # Transform to coprecessing frame and get the frame quaternions.
        # transition_times will be interpreted as the beginning and ending
        # times (respectively) to transition from using the coprecessing
        # frame to stopping rotation altogether. This can be helpful for
        # ensuring that the frame doesn't fluctuate wildly during late
        # ringdown. This transition_times are assumed to be defined with
        # respect to a peak amplitude time of zero, so you want to set
        # transition_times=None for PN.
        w.to_coprecessing_frame(transition_times=transition_times)
        qc = w.frame

    else:
        # If use_news=True, we use news instead of h to define the
        # coprecessing frame, because that lets us define the coprecessing
        # frame in a more supertranslation invariant way, especially in the
        # presence of memory.
        w_news = w.copy()
        w_news.data = w.data_dot
        w_news.dataType = scri.hdot
        w_news.to_coprecessing_frame(transition_times=transition_times)

        # Use the coprecessing frame quat obtained from the news to
        # transform the strain
        qc = w_news.frame
        w.rotate_decomposition_basis(qc)

    return quaternion.as_float_array(qc).T, w.data.T

def angularVelocityAndAmplitude(t, h):
    """ Same as omegaOrb_from_waveform, but with the right name.
    """
    return omegaOrb_from_waveform(t, h, return_amp=True)

def omegaOrb_from_waveform(t, h, return_amp=False):
    """ This should have been called angular_velocity, but it's
    called omegaOrb to match the old GWFrames function. So, this
    is NOT the frequency in the coprecessing frame.
    """

    w = scri.WaveformModes(
          dataType = scri.h,
          t = np.copy(t),
          data = np.copy(h.T),      # Because data gets modified internally
          ell_min = 2,
          ell_max = lMax_from_num_modes(len(h)),
          frameType = scri.Inertial,
          r_is_scaled_out = True,
          m_is_scaled_out = True,
    )

    # Use the angular_velocity so that there's less nutation in the data
    omegaOrbital = np.linalg.norm(w.angular_velocity(), axis=1)

    if return_amp:
        return omegaOrbital, w.norm(take_sqrt=True)
    else:
        return omegaOrbital

#-------------------------------------------------------------------------
# Helper functions for rotating waveform modes
def _floatFac(n):
    if n < 0:
        raise ValueError("Got negative argument to factorial")
    if n == 0:
        return 1.0
    return n*_floatFac(n-1)

def _floatFacRatio(n, k=0):
    """Ratio of n! to k! but only computes n*(n-1)*...*(n-k) for speed"""
    if n < 0:
        raise ValueError("Got negative argument to factorial")
    if n == k:
        return 1.0
    return n*_floatFacRatio(n-1, k=k)

def _binom(n, k):
    #return _floatFac(n)/(_floatFac(n-k)*_floatFac(k))
    return _floatFacRatio(n,k) / _floatFac(n-k)

def _wignerCoef(L, mp, m):
    return np.sqrt(_floatFac(L+m)*_floatFac(L-m) /
            (_floatFac(L+mp)*_floatFac(L-mp)))

def _wignerD(q, L, mp, m):
    """
(Code adapted to python from GWFrames)
    """
    if abs(mp) > L or abs(m) > L:
        raise ValueError("Bad indices")
    ra = q[0] + 1.j*q[3]
    rb = q[2] + 1.j*q[1]
    ra_small = (abs(ra) < 1.e-12)
    rb_small = (abs(rb) < 1.e-12)
    i1 = np.where((1 - ra_small)*(1 - rb_small))[0]
    i2 = np.where(ra_small)[0]
    i3 = np.where((1 - ra_small)*rb_small)[0]
    res = 0. * ra

    # Determine res at i2: it's 0 unless mp == -m
    if mp==(-m):
        if (L+mp)%2 == 0:
            res[i2] = rb**(2*m)
        else:
            res[i2] = rb**(2*m)

    # Determine res at i3: it's 0 unless mp == m
    if mp == m:
        res[i3] = ra**(2*m)

    # Determine res at i1, where we can safely divide by ra and rb
    ra = ra[i1]
    rb = rb[i1]

    absRRatioSquared = (abs(rb)/abs(ra))**2
    rhoMin = max(0, mp-m)
    rhoMax = min(L+mp, L-m)
    factor = _wignerCoef(L, mp, m)*(abs(ra)**(2*(L-m)))*(ra**(m+mp))*(rb**(m-mp))
    s = 0.
    for rho in range(rhoMax, rhoMin-1, -1):
        s = ((-1)**rho)*_binom(L+mp, rho)*_binom(L-mp, L-rho-m) + (s*absRRatioSquared)
    res[i1] = factor*s*(absRRatioSquared**rhoMin)
    return res

#-------------------------------------------------------------------------
def rotateWaveform(t, quat, h, inverse=0):
    """
Similar to transformWaveform but does not rely on GWFrames.
(Code adapted to python from GWFrames)
    """
    if not inverse:
        quat = quatInv(quat)
    res = 0.*h
    L=2
    i=0
    while i < len(h):
        for m in range(-L, L+1):
            for mp in range(-L, L+1):
                res[i+m+L] += _wignerD(quat, L, mp, m)*h[i+mp+L]
        i += 2*L + 1
        L += 1
    return res

def transformWaveform(t, quat, h, inverse=0):
    """Transforms a waveform h according to some frame expressed as unit
    quaternions.

    If h is in the coprecessing frame and quat is the coprecessing frame
    quaternion, using inverse=0 gives the waveform in the inertial frame.
    Use inverse=1 if h is in the inertial frame and you want the
    coprecessing frame waveform.
    """

    if inverse:
        quat = quatInv(quat)

    w = scri.WaveformModes(
          dataType = scri.h,
          t = np.copy(t),
          data = np.copy(h.T),      # Because data gets modified internally
          ell_min = 2,
          ell_max = lMax_from_num_modes(len(h)),
          frame = quaternion.as_quat_array(quat.T),
          frameType = scri.Inertial,
          r_is_scaled_out = True,
          m_is_scaled_out = True,
    )

    w.to_inertial_frame()

    return w.data.T

#-------------------------------------------------------------------------
def alignVec_quat(vec):
    """Returns a unit quaternion that will align vec with the z-axis"""
    alpha = np.arctan2(vec[1], vec[0])
    beta = np.arccos(vec[2])
    gamma = -alpha*vec[2]
    cb = np.cos(0.5*beta)
    sb = np.sin(0.5*beta)
    return np.array([cb*np.cos(0.5*(alpha + gamma)),
                     sb*np.sin(0.5*(gamma - alpha)),
                     sb*np.cos(0.5*(gamma - alpha)),
                     cb*np.sin(0.5*(alpha + gamma))])

#-------------------------------------------------------------------------
def zRotationQuat(phi):
    """Returns a unit quaternion that will rotate about the z-axis"""
    return np.array([np.cos(phi/2.), 0., 0., np.sin(phi/2.)])

#-------------------------------------------------------------------------
def lHat_from_quat(quat):
    qInv = quatInv(quat)
    return multiplyQuats(quat, multiplyQuats(
                    np.array([0., 0., 0., 1.]), qInv))[1:]

#-------------------------------------------------------------------------
def transformTimeDependentVector(quat, vec, inverse=0):
    """Given (for example) a minimal rotation frame quat, transforms
    vec from the minimal rotation frame to the inertial frame.
    With inverse=1, transforms from the inertial frame to the minimal
    rotation frame."""
    qInv = quatInv(quat)
    if inverse:
        return transformTimeDependentVector(qInv, vec, inverse=0)

    return multiplyQuats(quat, multiplyQuats(np.append(np.array([
            np.zeros(len(vec[0]))]), vec, 0), qInv))[1:]

#-------------------------------------------------------------------------
def simple_precession_alignment(t, h, iAlign):
    """
    Rotates h using constant-in-time rotations such that at t[iAlign],
    -The z-direction corresponds to maximal gravitational wave emission
    -The phases of the (2, 2) and (2, -2) modes are equal (and nearly zero)
    -The average of phases of the (2, 1) and (2, -1) modes is less than pi/2
     The last condition resolves an ambiguity of a rotation by pi about the
     z-axis.

    Note that this routine requires only waveforms, not horizon data.
    """

    quat = coprecessingQuat(t, h)
    lHat = lHat_from_quat(quat)
    align_quat = np.array([1., 0., 0., 0.])

    # Rotate the waveform such that lHat[iAlign] points in the z-direction
    zErr1 = np.sqrt(np.sum((lHat[:, iAlign] - np.array([0., 0., 1.])) ** 2))
    zErr2 = np.sqrt(np.sum((lHat[:, iAlign] + np.array([0., 0., 1.])) ** 2))
    if zErr2 < zErr1:
        lHat = -1 * lHat
        zErr = zErr2
    else:
        zErr = zErr1

    if zErr > 1.e-10:
        quat0 = alignVec_quat(lHat[:, iAlign])
        align_quat = multiplyQuats(align_quat, quat0)
        h_aligned = transformWaveform(t, np.array([quat0 for _ in t]).T, h, 1)
    else:
        h_aligned = 1. * h

    # Rotate about the z-axis such that the phases of the (2, 2) and (2, -2)
    # modes are ~ 0.  This leaves an ambiguity: we may rotate by an additional
    # pi radians.
    p22 = np.angle(h_aligned[4][iAlign])
    p2m2 = np.angle(h_aligned[0][iAlign])
    pOrb = 0.25 * (p22 - p2m2)
    p21 = np.angle(h_aligned[3][iAlign])
    p2m1 = np.angle(h_aligned[1][iAlign])
    # p21diff is the new phase angle of the 2,1 mode, restricted
    # between -pi and +pi.
    # If this is less than pi/2, then add pi to pOrb.
    p21diff = ((p21 - p2m1) / 2. - pOrb + np.pi) % (2 * np.pi) - np.pi
    if abs(p21diff) < np.pi / 2.:
        pOrb += np.pi
    if abs(pOrb) > 1.e-10:
        rotQuat = zRotationQuat(-pOrb)
        align_quat = multiplyQuats(align_quat, rotQuat)
        h_aligned = transformWaveform(t, np.array([rotQuat for _ in t]).T,
                                      h_aligned, 1)
    return h_aligned

#-------------------------------------------------------------------------
def precession_alignment(t, h, iAlign=0, useFilteredQuat=0,
            filterCycleWidth=0.25, returnQuat=0, returnAlignQuat=0,
            nearlyAligned=1,
            transition_times=COPR_TAPER_TRANSITION_TIMES_DEFAULT):
    """
Aligns h using constant-in-time rotations such that at t[iAlign]:
    -The z-direction corresponds to maximal gravitational wave emission
    -The phases of the (2, 2) and (2, -2) modes are equal (and nearly zero)
This leaves an ambiguity of an additional rotation by pi about the z-axis.
It is assumed the waveform is already nearly aligned, in which case we just
choose the smaller rotation.
If nearlyAligned=0, you might have issues.
Typically we need more than just the waveform to resolve this ambiguity,
i.e. we want the larger black hole roughly on the positive x-axis,
and the smaller one on the negative x-axis, with the orbital angular
momentum roughly in the z-direction.
See coprecessingQuatAndWaveform and top of the file for docs on transition_times
    """
    if useFilteredQuat:
        quat = filteredCoprQuat(t, h, widthCycles=filterCycleWidth)
    else:
        quat = coprecessingQuat(t, h, transition_times=transition_times)
    lHat = lHat_from_quat(quat)
    align_quat = np.array([1., 0., 0., 0.])

    # Rotate the waveform such that lHat[iAlign] points in the z-direction
    zErr = np.sqrt(np.sum((lHat[:,iAlign] - np.array([0., 0., 1.]))**2))
    if zErr > 0.1 and nearlyAligned:
        # Maybe we got it upside down!
        lHat = -1*lHat
        zErr = np.sqrt(np.sum((lHat[:,iAlign] - np.array([0., 0., 1.]))**2))
    if zErr > 0.1 and nearlyAligned:
        raise Exception('Large waveform misalignment, lHat is %s!'%(
                lHat[:,iAlign]))
    if zErr > 1.e-10:
        quat0 = alignVec_quat(lHat[:,iAlign])
        align_quat = multiplyQuats(align_quat, quat0)
        h_aligned = transformWaveform(t,
                                      np.array([quat0 for _ in t]).T,
                                      h,
                                      1)
        if useFilteredQuat:
            quat = filteredCoprQuat(t, h_aligned,
                                    widthCycles=filterCycleWidth)
        else:
            quat = coprecessingQuat(t, h_aligned,
                    transition_times=transition_times)
    else:
        print("Already aligned in the z-direction!")
        h_aligned = 1.*h

    # Rotate about the z-axis such that the phases of the (2, 2) and (2, -2)
    # modes are ~ 0.  This leaves an ambiguity: we may rotate by an additional
    # pi radians.
    # We assume the given waveform is nearly aligned, and choose the smaller rotation.
    p22 = np.angle(h_aligned[4][iAlign])
    p2m2 = np.angle(h_aligned[0][iAlign])

    # We define the orbital phase as phi_orb = (phi_2m2 - phi_22)/4.
    # pOrb is the orbital phase at iAlign.
    # pOrb of 0 or -pi/2 or pi/2 are ok.
    # Before the SpEC sign change, pOrb was always zero. After the sign change:
    # h22_new = -h22 = A22 np.exp(1j*phi_22) * np.exp(\pm 1j*np.pi), meaning
    # phi22_new = phi_22 \pm np.pi.
    # Similarly, phi2m2_new = phi_2m2 \pm np.pi.
    # So, phi_orb_new = phi_orb or phi_orb - pi/2 or phi_orb + pi/2

    # Also NOTE: The assumption here is that pOrb ~ orbital_phase, where
    # orbital_phase is defined using the BH trajectories. This should breakdown
    # for sufficiently eccentric binaries because (assuming nonprecessing for
    # the sake of argument) phi22 = 2 * orbital_phase is only accurate for
    # quasicirular binaries. In rough_alignment, we use orbital_phase to align
    # the waveform. Here, we are checking the alignment using pOrb (which
    # becomes phi22/2 for nonprecessing). So, at some sufficiently high ecc, as
    # long as enough orbits or time have elapsed from the start of the waveform
    # to iAlign, this difference should show up. For now, let's set the
    # threshold to 0.25 radians so we are aware of it when it happens. For
    # example it seems to happen for case Private/Ecc3dSur/009, for which
    # t[0]=-11754.9, and metadata ecc=0.4, when setting t[iAlign]=-7000.
    pOrb = 0.25*(p22 - p2m2)
    if abs(pOrb) > 0.25 and abs(abs(pOrb) - np.pi/2) > 0.25 and nearlyAligned:
        print (t[0], t[iAlign])
        raise Exception('Large waveform misalignment, pOrb is %s!'%(pOrb))


    if not nearlyAligned:
        print("WARNING: nearlyAligned=False option has not been verified "
              "after the SpEC sign change")
        p21 = np.angle(h_aligned[3][iAlign])
        if abs(p21 - pOrb - np.pi/2.) > np.pi/2.:
            pOrb += np.pi

    if abs(pOrb) > 1.e-10:
        rotQuat = zRotationQuat(-pOrb)
        align_quat = multiplyQuats(align_quat, rotQuat)
        h_aligned = transformWaveform(t,
                                      np.array([rotQuat for _ in t]).T,
                                      h_aligned,
                                      1)

        if useFilteredQuat and returnQuat:
            quat = filteredCoprQuat(t, h_aligned,
                                    widthCycles=filterCycleWidth)
        elif returnQuat:
            quat = coprecessingQuat(t, h_aligned,
                    transition_times=transition_times)
    else:
        print("Already aligned in phase!")

    if returnQuat:
        # The current quat at iAlign might contain some z-rotation.
        # We don't want the waveform at iAlign to be rotated at all -
        # the coprecessing frame should agree with the inertial frame at iAlign.
        # Note the order of multiplication is important!
        q0 = quat.T[iAlign]
        if max(abs(q0[1:3])) > 1.e-8:
            raise Exception('?? q0=%s should not have x or y components.'%q0)
        quat = multiplyQuats(quatInv(q0), quat)
        if returnAlignQuat:
            return h_aligned, quat, align_quat
        return h_aligned, quat
    if returnAlignQuat:
        return h_aligned, align_quat
    return h_aligned

#-------------------------------------------------------------------------
def filterQuat(quat, phase, widthCycles=0.25, maxWidths=4, nInterpPtsFactor=2,
               normalize=True):
    # Interpolate onto constant phase
    if not all(np.diff(phase) > 0.):
        raise Exception('Phases should be monotonically increasing')
    nPts = len(phase)*nInterpPtsFactor
    dp = (phase[-1] - phase[0])/nPts
    uniform_phase = np.linspace(phase[0], phase[-1], nPts)
    quat2 = np.array([_spline(phase, tmp, s=0)(uniform_phase) for tmp in quat])
    sigma = widthCycles*2.*np.pi/dp
    maxWidth = int(maxWidths*sigma)
    fltr = np.exp(-(1.*np.array(range(-maxWidth, maxWidth+1))**2./(2.*sigma*sigma)))
    fltr = fltr/sum(fltr)
    quatF = np.array([np.convolve(tmp, fltr, mode='same') for tmp in quat2]).T
    # Fix the ends
    for i in range(maxWidth):
        tmpFltr = fltr[maxWidth-i:maxWidth+i+1]
        tmpFltr = tmpFltr/sum(tmpFltr)
        quatF[i] = np.array([np.sum(tmpFltr*tmp[:2*i+1]) for tmp in quat2])
        quatF[nPts-1-i] = np.array([np.sum(tmpFltr*tmp[nPts-1-2*i:]) for tmp in quat2])
    quatF = quatF.T

    # Interpolate back onto the original phase sampling, and normalize
    interp_quat = np.array([_spline(uniform_phase, tmp, s=0)(phase) for tmp in quatF])
    if not normalize:
        return interp_quat

    return interp_quat/np.sqrt(np.sum(interp_quat**2, 0))

#-------------------------------------------------------------------------
def filteredCoprQuat(t, h, widthCycles=0.25, iAlign=None):
    """Returns a quaternion similar to the minimally rotating
    coprecessing quaternion, but with orbital timescale oscillations
    filtered out.  It IS minimally rotating (no z component),
    but is NOT exactly coprecessing.
    widthCycles controls the width of the gaussian filter in number of cycles.
    Specify iAlign to ensure the coprecessing frame is aligned at t[iAlign]."""
    quat = coprecessingQuat(t, h, iAlign=iAlign)
    h_copr = transformWaveform(t, quat, h, inverse=1)
    orbPhase = np.append(0., np.diff(t)*np.cumsum(omegaOrb_from_waveform(t, h)))
    filteredQuat = filterQuat(quat, orbPhase, widthCycles=widthCycles)
    h_filtered = transformWaveform(t, filteredQuat, h_copr, 0)
    quat = coprecessingQuat(t, h_filtered)
    if iAlign is not None:
        q0 = quat[:,iAlign]
        quat = multiplyQuats(quatInv(q0), quat) # This ordering preserves min-rot.
    return quat


#### Old GWFrames functions
###
####---------------------------------------------------------------------
###def lm_GWFrames(nModes):
###    lMax = lMax_from_num_modes(num_modes)
###    lm = np.array([[L, m] for L in range(2, lMax+1) for m in range(-L, L+1)],
###                  dtype=np.intc)
###    return lm
###
###def coprecessingQuatAndWaveform(t, h):
###    p = _Pool(1)
###    quat, wave = p.map(_coprecessingQuatAndWaveform, [(t, h)])[0]
###    p.close()
###    p.join()
###    return quat, wave
###
###
###def _coprecessingQuatAndWaveform(t_h):
###    t, h = t_h
###    w = GWFrames.Waveform()
###    w.SetFrameType(1)
###    w.SetFrame(np.array([]))
###    w.SetLM(lm_GWFrames(len(h)))
###    w.SetT(t)
###    w.SetData(h)
###    w.TransformToCoprecessingFrame()
###    quat = np.array([[a[0], a[1], a[2], a[3]] for a in w.Frame()]).T
###    return quat, w.Data()
###
###def corotatingQuat(t, h):
###    """Uses GWFrames to find the corotating frame of h"""
###    p = _Pool(1)
###    quat = p.map(_corotatingQuat, [(t, h)])[0]
###    p.close()
###    p.join()
###    return quat
###
###def _corotatingQuat(t_h):
###    t, h = t_h
###    w = GWFrames.Waveform()
###    w.SetFrameType(1)
###    w.SetFrame(np.array([]))
###    w.SetLM(lm_GWFrames(len(h)))
###    w.SetT(t)
###    w.SetData(h)
###    w.TransformToCorotatingFrame()
###    quat = np.array([[a[0], a[1], a[2], a[3]] for a in w.Frame()]).T
###    return quat
###
###def _angularVelocityAndAmplitude(t_h):
###    t, h = t_h
###    w = GWFrames.Waveform()
###    w.SetFrameType(1)
###    w.SetFrame(np.array([]))
###    w.SetLM(lm_GWFrames(len(h)))
###    w.SetT(t)
###    w.SetData(h)
###    return w.AngularVelocityVector(), w.Norm(True)
###
###def angularVelocityAndAmplitude(t, h):
###    """Uses GWFrames to find the angular velocity and amplitude of h"""
###    p = _Pool(1)
###    omega,amp = p.map(_angularVelocityAndAmplitude, [(t, h)])[0]
###    p.close()
###    p.join()
###    return omega,amp
###
###def transformWaveform(t, quat, h, inverse=0):
###    """
###Transforms a waveform h according to some frame expressed as unit quaternions.
###If h is in the coprecessing frame and quat is the coprecessing frame,
###using inverse=0 gives the waveform in the inertial frame.
###Use inverse=1 if h is in the inertial frame and you want the coprecessing
###waveform.
###    """
###    if inverse:
###        quat = quatInv(quat)
###    p = _Pool(1)
###    res = p.map(_transformWaveform, [(t, quat, h)])[0]
###    p.close()
###    p.join()
###    return res
###
###def _transformWaveform(t_quat_h):
###    t, quat, h = t_quat_h
###    w = GWFrames.Waveform()
###    w.SetFrameType(2)
###    w.SetLM(lm_GWFrames(len(h)))
###    w.SetT(t)
###    w.SetData(h)
###    w.SetFrame(np.array([GWFrames.Quaternions.Quaternion(
###            a[0], a[1], a[2], a[3])
###            for a in (quat/np.sqrt(np.sum(quat**2., 0))).T]))
###    w.TransformToInertialFrame()
###    return w.Data()
###
####-------------------------------------------------------------------------
###def omegaOrb_from_waveform(t, h):
###    """
###Transforms the waveform to the corotating frame and determines an orbital
###frequency using the frame rotation rate.
###    """
###    p = _Pool(1)
###    res = p.map(_omegaOrb_from_waveform, [(t, h)])[0]
###    p.close()
###    p.join()
###    return res
###
###def _omegaOrb_from_waveform(t_h):
###    t, h = t_h
###    # Use the corotating frame
###    # so that there's less nutation in the data
###    w = GWFrames.Waveform()
###    w.SetFrameType(1)
###    w.SetFrame(np.array([]))
###    w.SetLM(lm_GWFrames(len(h)))
###    w.SetT(t)
###    w.SetData(h)
###    w.TransformToCorotatingFrame()
###    quat = np.array([[a[0],a[1],a[2],a[3]] for a in w.Frame()])
###    dquat = multiplyQuats(quatInv(quat[:-1].transpose()),quat[1:].transpose())[1:]
###    omegaOrbital = 2.*np.sqrt(np.sum(dquat*dquat, 0))/np.diff(t)
###    return omegaOrbital
