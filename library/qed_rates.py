import numpy as np
from scipy.special import kv, zeta
from scipy.integrate import quad, quadrature
from scipy.constants import c, e, m_e, epsilon_0, mu_0, N_A, k, h, hbar, alpha, value

# Classical electron radius
r_e = value('classical electron radius')

# Conversion factor from J to MeV
fac = 1.e-6 * m_e * c**2 / e

def d2N_dt_dgg_quantum(gam_e,chi_e,gam_g):

    """
    Quantum differential rate of photon production by the Non Linear Inverse Compton scattering
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    gam_g : is the quantum parameter of the produced photon
    returns the rate in USI units in #photon / s
    """

    conditions = (gam_e > 1.0) and (gam_g < gam_e - 1.0)

    if conditions :

        Pcl = (2. * (m_e * c**2)**2 * alpha) / (3. * hbar)
        fac1 = Pcl * chi_e**2 / (gam_g * m_e * c**2)
        fac2 = -np.sqrt(3.) * gam_g / (2. * np.pi * gam_e**2 * chi_e**2)

        y = gam_g / (3. * chi_e * (gam_e - gam_g))

        def T1(y):
            integrand = lambda u : kv(1./3.,u)
            return quad(integrand, 2.*y, np.inf)[0]

        def T2(y):
            chi_g = gam_g * chi_e / gam_e
            return (2. + 3. * chi_g * y) * kv(2./3.,2.*y)

        result = fac1 * fac2 * (T1(y) - T2(y))

    else:

        result = 0.0

    return result

def d2N_dt_dgg_classic(gam_e,chi_e,gam_g):

    """
    Classical differential rate of photon production by the Non Linear Inverse Compton scattering
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    gam_g : is the quantum parameter of the produced photon
    returns the rate in USI units in #photon / s
    """

    conditions = (gam_e > 1.0) and (gam_g <= gam_e - 1.0)

    if conditions :

        Pcl = (2. * (m_e * c**2)**2 * alpha) / (3. * hbar)
        fac1 = Pcl * chi_e**2 / (gam_g * m_e * c**2)
        fac2 = 3. * np.sqrt(3.) / (4. * np.pi * chi_e**2)
        fac3 = chi_e / gam_e
        chi_g = gam_g * fac3
        y = 2. * chi_g / (3. * chi_e**2)

        integrand = lambda u : y*kv(5./3.,u)

        result = fac1 * fac2 * fac3 * quad(integrand, y, np.inf)[0]

    else:

        result = 0.0

    return result

def dN_dt_NLIC_quantum(gam_e, chi_e, min_b=0.0):

    """
    Total quantum rate of photon production by the Nonlinear Inverse Compton process (numerical integration)
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    min_bound : minimum bound of integration, the maximum is always gam_e - 1
    returns the rate in USI units in #photons / s
    """

    integrand = lambda s : d2N_dt_dgg_quantum(gam_e,chi_e,s)

    return quad( integrand, min_b , gam_e - 1. )[0]

def dN_dt_NLIC_classic(gam_e, chi_e, min_b=0.0):

    """
    Total classical rate of photon production by the Nonlinear Inverse Compton process (numerical integration)
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    min_bound : minimum bound of integration, the maximum is always gam_e - 1
    returns the rate in USI units in #photons / s
    """

    integrand = lambda s : d2N_dt_dgg_classic(gam_e,chi_e,s)

    return quad( integrand, min_b , gam_e - 1. )[0]

def dE_dt_NLIC_quantum(gam_e, chi_e, min_b=0.0):

    """
    Total quantum energy of photon production by the Nonlinear Inverse Compton process (numerical integration)
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    min_bound : minimum bound of integration, the maximum is always gam_e - 1
    returns the rate in USI units in #photons / s
    """

    integrand = lambda s : s * d2N_dt_dgg_quantum(gam_e,chi_e,s)

    return quad( integrand, min_b , gam_e - 1. )[0]

def dE_dt_NLIC_classic(gam_e, chi_e, min_b=0.0):

    """
    Total classical rate of photon production by the Nonlinear Inverse Compton process (numerical integration)
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    min_bound : minimum bound of integration, the maximum is always gam_e - 1
    returns the rate in USI units in #photons / s
    """

    integrand = lambda s : s * d2N_dt_dgg_classic(gam_e,chi_e,s)

    return quad( integrand, min_b , gam_e - 1. )[0]

def d2N_dchip_dt(gam_g,chi_g,chi_p):

    """
    Differential rate of pair production by the Breit Wheeler process
    gam_g : is the photon energy normalised in mc^2
    chi_g : is the quantum parameter of the incident photon
    chi_p : is the quantum parameter of the produced positron
    returns the rate in USI units in #positrons / s
    """

    conditions = (gam_g > 2.0) and (chi_p < chi_g) and (chi_p>chi_g/gam_g)

    if conditions :

        chi_e = chi_g - chi_p
        x = ( chi_g / ( chi_e * chi_p ) )**( 2. / 3. )

        fac1 = m_e * c**2 * chi_g * alpha / ( hbar * gam_g )
        fac2 = 1. / ( np.pi * np.sqrt( 3. ) * chi_g**2 )

        # First intermediate integral
        def T1(x):
            integrand = lambda s : np.sqrt(s) * kv( 1./3. , (2./3.) * s**(3./2.) )
            return quad( integrand , x , np.inf )[0]

        # Second intermediate integral
        def T2(x):
            return ( 2. - chi_g * x**(3./2.) ) * kv( 2./3. , (2./3.) * x**(3./2.) )

        # Boolean that ensures that chi_e belongs to the correct range
        # chi_e is the electron quantum parameter
        test = ( chi_e > chi_g / gam_g ) and ( chi_e < chi_g * ( 1. - 1. / gam_g ) )

        result = test * fac1 * fac2 * ( T1(x) - T2(x) )

    else:

        result = 0.0

    return result

def d2N_BW_Ritus(gam_g, chi_g):

    """
    Total rate of pair production by the Breit Wheeler process (numerical integration)
    gam_g : is the photon energy normalised in mc^2
    chi_g : is the quantum parameter of the incident photon
    returns the rate in USI units in #positrons / s
    """

    integrand = lambda s : d2N_dchip_dt(gam_g, chi_g, s)

    return quad( integrand, chi_g / gam_g , (gam_g - 1.) * chi_g / gam_g)[0]

def dN_dt_NLIC_classic_cdf(gam_e, chi_e, k, min_b):

    """
    Total classical rate of photon production by the Nonlinear Inverse Compton process (numerical integration)
    gam_e : is the electron energy normalised in mc^2
    chi_e : is the quantum parameter of the incident electron
    min_bound : minimum bound of integration, the maximum is always gam_e - 1
    returns the rate in USI units in #photons / s
    """

    integrand = lambda s : d2N_dt_dgg_classic(gam_e, chi_e, s)

    numerator = quad( integrand, min_b , k )[0]
    denominator = quad( integrand, min_b , gam_e - 1. )[0]

    return numerator / denominator

def interp_spec_pairQED(chi_g):

    """
    Total rate of pair production by the Breit Wheeler process (fit from osiris)
    gam_g : is the photon energy normalised in mc^2
    chi_g : is the quantum parameter of the incident photon
    returns the rate in USI units in #positrons / s
    """


    p_array = [  -4.683083764792118 ,  -1.091211988944788 , -4.309558261657857 ,
        4.491779676559683,   -1.107281703238826,   -0.461518946888616 ,  1.754341491184969 , -2.973628079492484 ,   4.567415354988209 ,
        -1.114863165044028,   -0.019692135577373 ,  0.199292591618420 , -0.774418835297167  , 3.062688187056846 , -0.696902875956990 ]
    p_array = np.array(p_array)


    if ( chi_g > 1000.0 ):
        rk1 = kv( 1./3. , 4.0 / ( 3.0 * chi_g ) )
        interp = np.sqrt(3.0) * np.pi * 0.16 * (rk1**2) 

    else:
        ltc = np.log10(chi_g)
        if ( chi_g < 1.0 ):
            n = 1 
        elif (chi_g < 10.0):
            n = 6
        else:
            n = 11

    # Fortran : the index of arrays starts at 1
    # Python : the index of arrays starts at 0
    n -= 1

    p = p_array[n]*(ltc**4) + p_array[n+1]*(ltc**3) + p_array[n+2]*(ltc**2) + p_array[n+3]*ltc + p_array[n+4] 
    interp = ( 10**p ) / chi_g

    return interp

def d2N_BW_osiri(gam_g, chi_g):

    interp = interp_spec_pairQED(chi_g)
    coef_QED = alpha * m_e * c**2 / (np.sqrt(3.) * np.pi * hbar)
    Wpair = coef_QED * interp / gam_g

    return Wpair

def  specpair(chi_g):

    specpair = -1.0

    return specpair

def specpair_fast(chi_g):

    # coefficients for 0.1 < chi_g < 10
    p1_array = np.array([ -0.000005225401859,     -0.000080299602413,     0.000140584960763,     0.009000197490255,     0.035872232367153, \
         -0.000512836744468,     -0.005836167050800,     -0.024756793004610,     -0.039904597410351,     -0.001077077386309, \
         -0.004136487977769,     -0.030318650457853,     -0.087607137642237,     -0.112531074265945,     -0.032921973247876,  \
          0.057693258982845,     0.179283783937442,     0.180705696041231,     0.041353652234700,     0.000475472998592,  \
         -0.145328538070807,     -0.233082359479153,     -0.124513341694971,     -0.055982996943532,     -0.010737979156167 ])
    p2_array = np.array([ -0.000017186457941,     -0.000501308272203,     -0.005835292277119,     -0.031158652183816,     -0.014223734979921,  \
          0.000017025291305,     -0.000183506798540,     -0.004884676164642,     -0.030445605558507,     -0.014858163874475,   \
          0.002544961647828,     0.013669710411411,     0.023407691578925,     -0.004987492811994,     -0.006377414300204,  \
          0.020533649186576,     0.088984339299603,     0.140104063967745,     0.074713915215919,     0.013935964256991,  \
         -0.793006417498808,     -1.555472861127672,     -1.115360068815503,     -0.354708337189465,     -0.041628012743737 ])
    p3_array = np.array([  0.000053676346116,     0.001342395538715,     0.011878135658543,     0.033049144144598,     -0.129859763407402,  \
          0.001864793520613,     0.022926907676793,     0.111050470233787,     0.241997090137700,     0.040713187936894,  \
          0.010915660757696,     0.081889940287695,     0.257794761863463,     0.407301837742367,     0.111748829824497,  \
         -0.094704410042950,     -0.259225069042258,     -0.155491759584177,     0.184605599105285,     0.066710295541102,  \
         -1.321702458506052,     -2.669062636880125,     -1.968481008170236,     -0.435526991491079,     -0.014677433853649 ])
    p4_array = np.array([ 0.000146287990639,     0.004170117816959,     0.047793530425434,     0.281487950503571,     0.891407062392472,  \
          0.002355625428871,     0.031357451125372,     0.177383183278795,     0.565709748126881,     1.133279049852064,   \
          0.008894669086513,     0.075669333596765,     0.291976904248376,     0.699564295108904,     1.192750654018492,   \
         -0.052884210441859,     -0.128784161738488,     0.037015124061894,     0.557408165404765,     1.162815353169648,  \
         -0.626653322990238,     -1.231597890005989,     -0.776953017271845,     0.283251642562608,     1.127225781030429 ])
    p5_array = np.array([ 0.000101035895204,     0.003034961791079,     0.038517000340175,     0.296395206350618,     -0.340255742643951,  \
          0.001367534920433,     0.018706596307303,     0.113713739198754,     0.462593023080581,     -0.197612683567629,  \
          0.007675199160302,     0.058921747887693,     0.211874594432208,     0.571362244498481,     -0.151477476660753,  \
          0.036908328292682,     0.162109824654076,     0.350672986287757,     0.655784776438876,     -0.131875713605264,  \
          0.421923148153023,     0.869248252496938,     0.845970255776770,     0.813262278773464,     -0.112629271074978 ])

    # coefficients for 10 < chi_g < 100
    p6_array = np.array([ 0.000006082040645,     0.000564737585995,     0.008242128861585,     0.046025770183968,     0.084647285368009,  \
         -0.002485658336182,     -0.021312822865797,     -0.052708137449931,     0.005176422736026,     0.122282126102417,  \
         -0.029376830591882,     -0.213113658843963,     -0.568537193360927,     -0.614609623468511,     -0.158288550851653,  \
          0.111656465232389,     0.257195253365006,     0.026946304622848,     -0.274468967945381,     -0.084172472851054,   \
         -0.810417539310961,     -0.880137768795069,     -0.281075095233155,     -0.188412694151709,     -0.048622693553012 ])
    p7_array = np.array([ 0.000190332705603,     0.004331676342178,     0.041328744292588,     0.203626212858701,     0.533900410274208,  \
          0.010500649205440,     0.111418057985674,     0.453780672610153,     0.897182903247248,     0.958322396366310,  \
          0.067311632302158,     0.525103861335212,     1.586743785301362,     2.280266329245493,     1.593141572200228,  \
         -0.109511244628452,     -0.038419575312760,     0.903676736272272,     1.904879603203579,     1.513766659274223,  \
         -5.524238464421281,     -11.990343682335268,     -8.993570148855301,     -1.742244248896068,     1.008817566301562 ])
    p8_array = np.array([ 0.000081962355663,     0.003080647795245,     0.043137842803378,     0.340288131821678,     -0.174328302877703,  \
         -0.003003229443531,     -0.023844138917779,     -0.030457433806767,     0.297061696048246,     -0.117978287081051,  \
         -0.012253448661188,     -0.114401777093548,     -0.328095829070981,     -0.113331976962520,     -0.322994083205383,  \
         -0.037353987824001,     -0.187562068001494,     -0.396651342055515,     -0.131745222357037,     -0.320882921260851,   \
          3.978736012514611,     8.264364245327915,     6.286822482443694,     2.223401647058546,     -0.008653167964833 ])

    mirror = False

    ratiomin = 1.e-8
    ratiomax = 1.0-1.e-8
    if ( chi_g < 0.1 ) or ( chi_g > 100.0 ):
        eta = specpair(chi_g)
    else:
        ratio = np.random.uniform()

        if ratio > 0.5 :
            mirror = True
            ratio = 1.0 - ratio

        lt_c = np.log10(chi_g)

        if ( ratio < ratiomin ):
            ratio = ratiomin
        elif ( ratio > ratiomax ):
            ratio = ratiomax

        if ( ratio < 0.007 ):
            n = 1
        elif ( ratio < 0.025 ):
            n = 6
        elif ( ratio < 0.12 ):
            n = 11
        elif (ratio < 0.3):
            n = 16
        else:
            n = 21


        # We add this line because indexes begin at 0 in Python and 1 in Fortran
        n -= 1

        if 	(lt_c < 1.0): 
            pc1 = p1_array[n]   * ( lt_c**4.0 ) + p2_array[n]   * ( lt_c**3.0 ) + p3_array[n]   * ( lt_c**2.0 ) + p4_array[n]   * lt_c + p5_array[n] 
            pc2 = p1_array[n+1] * ( lt_c**4.0 ) + p2_array[n+1] * ( lt_c**3.0 ) + p3_array[n+1] * ( lt_c**2.0 ) + p4_array[n+1] * lt_c + p5_array[n+1] 
            pc3 = p1_array[n+2] * ( lt_c**4.0 ) + p2_array[n+2] * ( lt_c**3.0 ) + p3_array[n+2] * ( lt_c**2.0 ) + p4_array[n+2] * lt_c + p5_array[n+2] 
            pc4 = p1_array[n+3] * ( lt_c**4.0 ) + p2_array[n+3] * ( lt_c**3.0 ) + p3_array[n+3] * ( lt_c**2.0 ) + p4_array[n+3] * lt_c + p5_array[n+3] 
            pc5 = p1_array[n+4] * ( lt_c**4.0 ) + p2_array[n+4] * ( lt_c**3.0 ) + p3_array[n+4] * ( lt_c**2.0 ) + p4_array[n+4] * lt_c + p5_array[n+4] 
            
            ltr = np.log10(ratio)
            pc = pc1 * ( ltr**4.0 ) + pc2 * ( ltr**3.0 ) + pc3 * ( ltr**2.0 ) + pc4 * ltr + pc5

        else:
            pc1 = p6_array[n]   * ( lt_c**2.0 ) + p7_array[n]   * lt_c + p8_array[n] 
            pc2 = p6_array[n+1] * ( lt_c**2.0 ) + p7_array[n+1] * lt_c + p8_array[n+1] 
            pc3 = p6_array[n+2] * ( lt_c**2.0 ) + p7_array[n+2] * lt_c + p8_array[n+2] 
            pc4 = p6_array[n+3] * ( lt_c**2.0 ) + p7_array[n+3] * lt_c + p8_array[n+3] 
            pc5 = p6_array[n+4] * ( lt_c**2.0 ) + p7_array[n+4] * lt_c + p8_array[n+4] 
            
            ltr = np.log10(ratio)
            pc = pc1 * ( ltr**4.0 ) + pc2 * ( ltr**3.0 ) + pc3 * ( ltr**2.0 ) + pc4 * ltr + pc5

        eta = 10**pc

    if mirror :
        specpair_fast = chi_g - eta
    else:
        specpair_fast = eta

    return specpair_fast

def d2N_BW_Erber(gam_g, chi_g, precise=True):

    """
    Total rate of pair production by the Breit Wheeler process (theoretical integration)
    gam_g : is the photon energy normalised in mc^2
    chi_g : is the quantum parameter of the incident photon
    returns the rate in USI units in #positrons / s
    """

    TnBW = 0.16 * kv( 1./3. , 4. / (3. * chi_g) )**2 / chi_g
    fac = m_e * c**2 * chi_g * alpha / (hbar * gam_g)

    if precise :
        TnBW /= ( 1.0 - 0.172 / ( 1. + 0.295 * chi_g**(2./3.) ) )

    return fac * TnBW

def ltf(Z):
    '''
    This function determines the Thomas Fermi length for a given atomic number Z
    inputs :
        Z is th atomic number
    outputs :
        result is the TF length, normalised by the compton wavelength
    '''

    compton_wavelength = hbar / (m_e * c)
    length = (4. * np.pi * epsilon_0 * hbar **
              2 / (m_e * e ** 2) * Z ** (-1./3.))
    result = 0.885 * length / compton_wavelength

    return result


def I_1(d, l, q=1.0):
    '''
    This function computes the term I1 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screened potential
    outputs :
        result is the term I1
    '''

    T1 = l * d * (np.arctan(l * d) - np.arctan(l))
    T2 = - (l ** 2 / 2.) * (1. - d) ** 2 / (1. + l ** 2)
    T3 = (1. / 2.) * np.log((1. + l ** 2.) / (1. + (l * d) ** 2))

    result = q ** 2 * (T1 + T2 + T3)  # why q?

    return result

def I_2(d, l, q=1.0):
    '''
    This function computes the term I2 in the Bremsstrahlung and Bethe-Heitler cross-section
    inputs :
        d is the delta parameter
        l is the screening length
        q is a fraction between 0 and 1 for this screenedpotential
    outputs :
        result is the term I2
    '''

    T1 = 4. * (l * d) ** 3 * (np.arctan(d * l) - np.arctan(l))
    T2 = (1. + 3. * (l * d) ** 2) * np.log((1. + l ** 2) / (1. + (l * d) ** 2))
    T3 = (6. * l ** 4 * d ** 2) * np.log(d) / (1. + l ** 2)
    T4 = l ** 2 * (d - 1.) * (d + 1. - 4. * l ** 2 * d ** 2) / (1. + l ** 2)

    result = 0.5 * q * (T1 + T2 + T3 + T4)

    return result


def bh_cs_dif(gp, k, Z):
    '''
    This function computes the differential Bethe-Heitler cross-section
    inputs :
        gp is the energy of the positron
        k is the energy of the photon
        Z is the atomic number
    outputs :
        result is the differential cross-section in m^2
    '''

    result = 0.0
    condition = (k >= 2.) and (gp >= 1.) and (
        gp <= k-1.)

    if condition:

        q = 1.0
        ge = k - gp
        d = k / (2.0 * gp * ge)
        l = ltf(Z)

        # Coulomb correction term
        fc = 0.0
        if k > 200. :
            fc = (alpha * Z) ** 2 / (1. + (alpha * Z) ** 2)
            sum = 0.0
            for n in range(1, 5):
                sum = sum + ((-alpha * Z) ** 2) ** n * (zeta(2. * n + 1) - 1)
            fc = fc*sum

        T1 = 4. * (Z * r_e) ** 2 * alpha / k ** 3
        T2 = (gp ** 2 + ge ** 2) * (I_1(d, l, q) + 1.0 - fc)
        T3 = (2. / 3.) * gp * ge * (I_2(d, l, q) + 5. / 6. - fc)

        result = T1 * (T2 + T3)

    return result


def bh_cs(Z, k):
    '''
    This function computes the total Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
    outputs :
        result is the total cross-section in m^2
    '''

    result = 0.0
    condition = (k > 2.0)

    if condition:
        result = quad(bh_cs_dif, 1.0, k-1.0, args=(k, Z))[0]

    return result


def bh_cdf(Z, k, gp):
    '''
    This function computes the CDF of the Bethe-Heitler cross-section
    inputs :
        Z is the atomic number
        k is the energy of the photon
        gp is the energy of the positron
    outputs :
        result is the CDF of the Bethe-Heitler cs (no units and between 0 and 1 by definition)
    '''

    condition = (k >= 2.) and (gp >= 0.) and (gp <= 1.)

    result = 0.0
    if condition:
        gp = 1.0 + (k - 2.) * gp
        numerator = quad(bh_cs_dif, 1.0, gp, args=(k, Z))[0]
        denominator = bh_cs(Z, k)
        result = numerator / denominator

    return result


def nr_cs_dif(Z, k, g1):
    '''
    This function computes the differential Bremsstrahlung cross-section for the nonrelativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the nonrelativistic case
    '''

    condition = (k > 0.) and (k < g1 - 1.)
    # Photon energy is bigger than zero and cannot be bigger than the sum between the energy of the electron it was emmited and its rest energy (normalized!)

    result = 0.0
    # in case condition fails

    if condition:

        g2 = g1 - k

        p1 = np.sqrt(g1 ** 2 - 1.)  # normalized momentum

        p2 = np.sqrt(g2 ** 2 - 1.)

        # ratio between velocity and light speed
        b1 = np.sqrt(1.-(1. / g1 ** 2))

        b2 = np.sqrt(1.-(1. / g2 ** 2))  # ratio after emitting photon

        dplus = p1 + p2  # maximum momentum

        dminus = p1 - p2  # minimum momentum

        L = ltf(Z)  # Thomas - Fermi Length

        T1 = 16. * (Z * r_e) ** 2 * alpha / (3. * k * p1 ** 2)  # factor

        T2 = (1. / 2.)*(np.log(((dplus * L) ** 2 + 1.) / ((dminus * L) ** 2 + 1.)) +
                        (1. / ((dplus * L) ** 2 + 1.)) - (1. / ((dminus * L) ** 2 + 1.)))  # glf

        T3 = 1.0
        if Z*alpha*(1./b2 - 1./b1) < 0.01 :
            T3 = (b1 / b2) * ((1. - np.exp(-2. * np.pi * Z * alpha / b1)) / (1. - np.exp(-2. * np.pi * Z * alpha / b2)))  # Elwert correction factor

        result = T1 * (T2) * T3
    return result


def mr_cs_dif(Z, k, g1):
    '''
    This function computes the differential Bremsstrahlung cross-section for the moderately relativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the moderately relativistic case
    '''

    condition = (k > 0.) and (k < g1 - 1.)

    result = 0.0

    if condition:

        d = k / (2. * g1 * (g1-k))  # momentum transfer

        l = ltf(Z)  # thomas-fermi length

        q = 1.0  # q ratio

        T1 = 4. * (Z * r_e) ** 2 * alpha / k  # factor

        T2 = (1. + ((g1 - k) / g1) ** 2) * (I_1(d, l, q) + 1.)

        T3 = (2. / 3.)*(((g1 - k) / g1)) * (I_2(d, l, q) + (5. / 6.))

        result = T1 * (T2 - T3)

    return result


def ur_cs_dif(Z, k, g1):
    '''
    This function computes the differential Bremsstrahlung cross-section for the moderately relativistic case
    inputs :
        Z is the atomic number
        k is the energy of the photon
        g1 is the energy of the electron
    outputs :
        differential Bremsstrahlung cross-section for the moderately relativistic case
    '''

    condition = (k > 0.) and (k < g1 - 1.)

    result = 0.0

    if condition:

        # initialization of Coulomb correction term
        fc = (alpha * Z) ** 2 / (1. + (alpha * Z) ** 2)

        sum = 0.0

        # loop for zeta function
        for n in range(1, 5):
            sum = sum + ((-alpha * Z) ** 2) ** n * (zeta(2. * n + 1) - 1)

        fc = fc*sum

        d = k / (2. * g1 * (g1 - k))  # momentum transfer

        l = ltf(Z)  # thomas-fermi length

        q = 1.0

        T1 = 4. * (Z * r_e) ** 2 * alpha / k

        T2 = (1.0 + ((g1 - k)/g1) ** 2) * (I_1(d, l, q) + 1.0 - fc)

        T3 = (2. / 3.)*(((g1 - k)/g1)) * (I_2(d, l, q) + (5. / 6.) - fc)

        result = T1 * (T2 - T3)

    return result

def br_cs_dif(Z, k, g1):

    if g1 <= 2:
        return nr_cs_dif(Z, k, g1)

    elif 2 < g1 <= 100:
        return mr_cs_dif(Z, k, g1)

    else:
        return ur_cs_dif(Z, k, g1)

def regime(ge):
    if ge <= 2:
        function = nr_cs_dif

    elif 2 < ge <= 100:
        function = mr_cs_dif

    else:
        function = ur_cs_dif

    return function


def br_cs(Z, ge, mode=None):
    '''
    This function computes the Bremsstrahlung cross-section
    inputs :
        df is differential Bremsstrahlung cross-section function
        Z is the atomic number
        g1 is the energy of the electron
        mode makes multiply
    outputs :
        Bremsstrahlung cross-section
    '''

    ge_m1 = ge - 1

    df = regime(ge)

    if mode == "k":
        result = quadrature(lambda k, Z, ge: k * df(Z, k, ge),
                            0., ge_m1, args=(Z, ge), vec_func=False)[0]
    else:
        result = quadrature(lambda k, Z, ge: df(Z, k, ge), 0., ge_m1,
                            args=(Z, ge), vec_func=False)[0]

    return result

def br_cdf_gauleg(Z, k, ge, if_log):
    '''
    This function computes the CDF of the Bremsstrahlung cross-section
    inputs :
        df is the density function to be integrated
        Z is the atomic number
        k is photon energy
        g1 is the energy of the electron
        mode makes multiply
    outputs :
        CDF value
    '''

    if if_log :

        g = lambda x: x * br_cs_dif(Z, x, ge)
        deg = 100

        # Numerator
        a = np.log(1.e-9*(ge-1.))
        b = np.log(k)

        x, w = np.polynomial.legendre.leggauss(deg)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        numerator = 0.0
        for j in range(deg):
            numerator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

        # Denominator
        a = np.log(1.e-9*(ge-1.))
        b = np.log(ge-1.)

        x, w = np.polynomial.legendre.leggauss(deg)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        denominator = 0.0
        for j in range(deg):
            denominator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

        result = numerator / denominator

    else :

        f = lambda x: br_cs_dif(Z, x, ge)
        deg = 100

        # Numerator
        a = 0.0
        b = k

        x, w = np.polynomial.legendre.leggauss(deg)
        t = 0.5 * (x + 1) * (b - a) + a

        numerator = 0.0
        for j in range(deg):
            numerator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

        # Denominator
        a = 0.0
        b = ge - 1.0

        x, w = np.polynomial.legendre.leggauss(deg)
        t = 0.5 * (x + 1) * (b - a) + a

        denominator = 0.0
        for j in range(deg):
            denominator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

        result = numerator / denominator

    return result

def ct_cs_tot(Z, g1):

    # Coulomb Trident process, Total cross-section 
    # This is a fit using different intervals

    result = 0.0

    if (g1 > 3.0 and g1 < 200 ):
        T0 = (g1 - 1.0) * fac
        result = 1.e-34 * 5.22 * Z**2 * np.log((2.3 + T0) / 3.52) **3
    else:
        result = ct_cs_tot_lim2(Z, g1)

    return result

def ct_cs_tot_lim1(Z, g1):
    # Coulomb Trident process, Total cross-section in the low energy limit
    return (7.0 / 2304.) * (Z * r_e * alpha)**2 * (g1 - 3.)**3

def ct_cs_tot_lim2(Z, g1):
    # Coulomb Trident process, Total cross-section in the high energy limit
    return (28.0 / (27.*np.pi) ) * (Z * r_e * alpha)**2 * (np.log(g1))**3

def ct_cs_diff_cdf(Z, g1, gpa, method='gauleg_log'):

    numerator = 0.0
    denominator = 0.0

    offset = 0.1
    min_gpa, max_gpa = 2.0, g1-1.0

    # For Gauss Legendre integration
    deg = 100
    x, w = np.polynomial.legendre.leggauss(deg)

    if method == 'gauleg_log' :

        # Define bounds and function to integrate
        a = np.log(min_gpa)
        b = np.log(gpa)
        g = lambda dummy: dummy * ct_cs_dif(Z, g1, dummy)

        # Rescale the Legendre points from the default bounds (-1,1) to our bounds (a,b)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        for j in range(deg):
            numerator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

        # Define bounds and function to integrate
        a = np.log(min_gpa)
        b = np.log(max_gpa)

        # Rescale the Legendre points from the default bounds (-1,1) to our bounds (a,b)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)

        for j in range(deg):
            denominator += np.sum(w[j] * g(t[j])) * 0.5 * (b - a)

    elif method =='gauleg_lin':

        # Define bounds and function to integrate
        a = min_gpa
        b = gpa
        f = lambda dummy: ct_cs_dif(Z, g1, dummy)

        # Rescale the Legendre points from the default bounds (-1,1) to our bounds (a,b)
        t = 0.5 * (x + 1) * (b - a) + a
        
        # Perform the integration
        for j in range(deg):
            numerator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

        # Define bounds and function to integrate
        a = min_gpa
        b = max_gpa

        # Rescale the Legendre points from the default bounds (-1,1) to our bounds (a,b)
        t = 0.5 * (x + 1) * (b - a) + a
        
        # Perform the integration
        for j in range(deg):
            denominator += np.sum(w[j] * f(t[j])) * 0.5 * (b - a)

    elif method == 'quad':

        # Define bounds and function to integrate
        a = min_gpa
        b = gpa
        f = lambda dummy: ct_cs_dif(Z, g1, dummy)

        # Perform the integration
        numerator, error = quad(f, a, b)

        # Define bounds and function to integrate
        a = min_gpa
        b = max_gpa

        # Perform the integration
        denominator, error = quad(f, a, b)

    return numerator / denominator
    
def ct_cs_dif(Z, g1, gpa):

    # Coulomb Trident process, differential cross-section in pair energy
    # This is a harmonic mean between two regimes

    result = 0.0

    if ((gpa > 2.) and (gpa < g1 - 1.0)) :

        ct_cs_dif_lim_low_gpa = ct_cs_dif_lim1(Z, g1, gpa)
        ct_cs_dif_lim_hig_gpa = ct_cs_dif_lim2(Z, g1, gpa)
        result = ct_cs_dif_lim_low_gpa * ct_cs_dif_lim_hig_gpa / (ct_cs_dif_lim_low_gpa + ct_cs_dif_lim_hig_gpa)

    return result

def ct_cs_dif_lim1(Z, g1, g_pa):

    # Coulomb Trident process, differential cross-section in pair energy
    # in the limit of low pair energy

    # T_pa is the sum of electron + positron kinetic energy, in mc^2
    T_pa = (g_pa - 2.0) * m_e * c**2

    x = 1.0 / g1
    C = 4.0 * (x**2 / (1.0-x**2)) * np.log(1.0 / x**2) - 4.0 * x**2 / 3.0 + x**4 / 6.0
    Cz = 3.0 * (x**2 / (1.0-x**2)) * (1.0 - (x**2 / (1.0-x**2)) * np.log(1.0 / x**2)) - 13.0 * x**2 / 5.0 + 7.0 * x**4 / 4.0 - 9.0 * x**6 / 10.0 + x**8 / 5.0
    Cr = -(3.0 / 2.0) * (x**2 / (1.0-x**2)) * (1.0 - (x**2 / (1.0-x**2)) * np.log(1.0 / x**2)) + 4.0 * x**2 / 5.0 - x**4 / 8.0 - x**6 / 20.0 + x**8 / 40.0

    # m^2/J
    result = (Z * r_e * alpha)**2 / 32.0 * (np.log(g1**2) - 161./60. + C + Cr + Cz) * T_pa**3 / (m_e * c**2)**4

    # m^2
    result *= m_e * c**2

    return result

def ct_cs_dif_lim2(Z, g1, g_pa):

    # Coulomb Trident process, differential cross-section in pair energy
    # in the limit of high pair energy

    # E_pa is the sum of electron + positron total energy, in mc^2
    E_pa = g_pa * m_e * c**2
    
    C1 = 1.0
    C2 = 1.0

    # m^2/J
    result = (56. / (9. * np.pi)) * (Z * r_e * alpha)**2 * np.log(C1 * E_pa / (m_e * c**2)) * np.log(C2 * m_e * c**2 * g1 / E_pa) / E_pa

    # m^2
    result *= m_e * c**2

    return result

def ct_cs_ddif(Z, g1, gp, ge):

    # Coulomb Trident process, differential cross-section in electron and positron energy
    # in the limit of high pair energy

    result = 0.0

    # Conditions on the bounds of each particle
    gpa = gp + ge
    bounds_correct = (g1 >= gpa + 1.0) and (gp >= 1.0) and (ge >= 1.0)

    # Condition to avoid negative values of term T2
    condition = (gpa > 4.0)
    if condition:

        lim_low = 0.5 * gpa * (1.0 - np.sqrt(1.0 - 4.0/ gpa))
        lim_hig = 0.5 * gpa * (1.0 + np.sqrt(1.0 - 4.0/ gpa))
        condition = (gpa > 4.0) and (ge > lim_low) and (ge < lim_hig)

    # If all condition are met, we can compute the cross-section
    if condition and bounds_correct:

        C1 = 1.0
        C2 = 1.0

        E_1 = g1 * m_e * c**2
        E_e = ge * m_e * c**2
        E_p = gp * m_e * c**2

        T0 = (8. / np.pi) * (Z * r_e * alpha)**2
        T1 = (E_p**2 + E_e**2 + 2.0 * E_p * E_e / 3.0) * (m_e * c**2)**2 / (E_e + E_p)**4
        T2 = np.log(C1 * E_e * E_p / ((E_e + E_p) * m_e * c**2))
        T3 = np.log(C2 * E_1 / (E_e + E_p))

        if T1<0.:
            print('T1<0')
        if T2<0.:
            print('T2<0')
        if T3<0.:
            print('T3<0')

        result =  T0 * T1 * T2 * T3

    return result

def ct_cs_ddif_generic(gpa, gp):

    # Coulomb Trident process, differential cross-section in electron and positron energy
    # in the limit of high pair energy
    ge = gpa - gp
    result = (gp**2 + ge**2 + 2.*gp*ge/3.) * np.log(gp*ge/gpa)

    return result

def ct_cs_ddif_cdf_generic(gpa, gp):

    # Function to integrate
    f = lambda dummy: ct_cs_ddif_generic(gpa, dummy)

    # numerator
    a, b = 1.0, gp
    numerator, error = quad(f, a, b)

    # denominator
    a, b = 1.0, gpa - 1.0
    denominator, error = quad(f, a, b)

    result = numerator / denominator

    return result

def ct_cs_ddiff_cdf(Z, g1, gpa, gp, method='gauleg_log'):

    result = 0.0
    min_gp, max_gp = 1.0, gpa - 1.0

    numerator, denominator = 0.0, 0.0

    # For Gauss Legendre integration
    deg = 100
    x, w = np.polynomial.legendre.leggauss(deg)

    if method == 'quad':

        # Function to integrate
        f = lambda dummy: ct_cs_ddif(Z, g1, gpa - dummy, dummy)

        # numerator
        a, b = 1.0, gp
        numerator, error = quad(f, a, b)

        # denominator
        a, b = 1.0, max_gp
        denominator, error = quad(f, a, b)

        result = numerator / denominator

    #------------------------------------------------
    # Gauleg_lin
    #------------------------------------------------

    elif method=='gauleg_lin':

        # Function to integrate
        f = lambda dummy: ct_cs_ddif(Z, g1, gpa - dummy, dummy)

        # numerator
        a, b = 1.0, gp
        t = 0.5 * (x + 1) * (b - a) + a
        for i_deg in range(deg):
            numerator += np.sum(w[i_deg] * f(t[i_deg])) * 0.5 * (b - a)

        # denominator
        a, b = 1.0, max_gp
        t = 0.5 * (x + 1) * (b - a) + a
        for i_deg in range(deg):
            denominator += np.sum(w[i_deg] * f(t[i_deg])) * 0.5 * (b - a)

        result = numerator / denominator

    elif method=='gauleg_log':

        # Function to integrate
        g = lambda dummy: dummy * ct_cs_ddif(Z, g1, gpa - dummy, dummy)

        # Define bounds and function to integrate
        a, b = np.log(1.0), np.log(gp)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)
        for i_deg in range(deg):
            numerator += np.sum(w[i_deg] * g(t[i_deg])) * 0.5 * (b - a)

        # Define bounds and function to integrate
        a, b = np.log(1.0), np.log(max_gp)
        t = np.exp(0.5 * (x + 1) * (b - a) + a)
        for i_deg in range(deg):
            denominator += np.sum(w[i_deg] * g(t[i_deg])) * 0.5 * (b - a)

        result = numerator / denominator

    return result

def fit_numpy_1D(x, f, degree, iflogx, iflogf, mode):

    '''
    This function uses numpy to fit a polynomial from the input data x, f
    inputs :
        x is the range of the variable
        f is the evaluation o the function to fit for x
        degree is the degree of the poynomial
        iflogx, iflogf are booleans to log x and f
        mode is a debug variable to test the polynomial reconstruction.
            It is 'auto' for a scipy evaluation and 'hand' for a reconstruction by hand
    '''

    # Log the data
    if iflogx :
        x = np.log10(x)
    if iflogf :
        f = np.log10(f)

    # Fit with numpy
    poly = np.polynomial.Polynomial.fit(x, f, deg=int(degree))
    coefs = poly.convert().coef

    # Build the result
    result = 0.0
    if mode == 'auto' :
        result = poly(x)

    elif mode == 'hand' :
        for i in range(int(degree+1)):
            result += coefs[i] * x ** i

    # Exponent the result
    if iflogf :
        result = 10 ** result

    return result, coefs