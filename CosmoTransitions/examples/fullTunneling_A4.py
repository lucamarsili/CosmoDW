import numpy as np
import matplotlib.pyplot as plt
import scipy
from cosmoTransitions import pathDeformation as pd


class Potential:
    """
    A sample potential. The depth of the absolute minimum is controlled with
    the parameters `fx` and `fy`.

    This potential has no physical significance whatsoever.
    """
    def __init__(self, mu2, g1, g2,A):
        #    define input parameters for potential, including dimension of scalar field (3 in this case)
        self._N_VEVs = 3
        
        self._mu2 = mu2
        self._g1 = g1
        self._g2 = g2
        self._A = A

        

        
    #       self.set_potential_parameters()
       
    #    construct scalar potential
    
    def V(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) * (phi1sq + phi2sq + phi3sq)/ 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0
        tmp +=  self._A  * (phi1 * phi2 * phi3)

        return tmp * scale



    #    construct first derivative of scalar potential, wrt phi1, phi2, phi3 NB ask about scale^3
    
    def Vtotal(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) ** 2 / 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0

        return tmp * scale
    def Vatz(self, phi1, phi2, phi3,i): 
        phi1sq = phi1[i]**2
        phi2sq = phi2[i]**2
        phi3sq = phi3[i]**2
       
        tmp = 0
        tmp += -self._mu2 *(phi1sq + phi2sq + phi3sq) / 2.0
        tmp +=  self._g1 * (phi1sq + phi2sq + phi3sq) ** 2 / 4.0
        tmp +=  self._g2 * (phi1sq * phi2sq + phi2sq * phi3sq + phi3sq * phi1sq) / 2.0
        return tmp

    def dV(self, field_values, scale=1.0):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        
        phi1sq = phi1**2
        phi2sq = phi2**2
        phi3sq = phi3**2

        # Initializing the result vector
        res = np.zeros(self._N_VEVs)
        
        # Calculating the derivatives
        dVdx = phi1 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi2sq + phi3sq) + self._A * ( phi2 * phi3 ))
        dVdy = phi2 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi3sq) + self._A * ( phi3 * phi1 ) )
        dVdz = phi3 * (-self._mu2 + self._g1 * (phi1sq +  phi2sq + phi3sq) + self._g2 * (phi1sq + phi2sq) + self._A * ( phi1 * phi2 ))
        
        rval = np.empty_like(field_values)
        rval[...,0] = dVdx
        rval[...,1] = dVdy
        rval[...,2] = dVdz
        
        #print("rval shape", rval.shape)
        # Scaling the result
        return rval
    #define function for tension, total energy, delta and scan
   
    # def GetTension(self, z, field_values, beta): 
    #     phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
    #     dphi3sqdz = [((((phi3[i+1] - phi3[i])/(z[i+1] - z[i]))**2) +(((phi2[i+1] - phi2[i])/(z[i+1] - z[i]))**2)+ (((phi1[i+1] - phi1[i])/(z[i+1] - z[i]))**2)) for i in range(len(phi3) - 1)]
    #     dphi3sqdz.append(0)
    #     pot1 = [ (phi2[i]**2 + phi1[i]**2 + phi3[i]**2 -1)**2 for i in range(len(phi3))]
    #     pot2 = [ beta*(phi1[i]**2)*(phi3[i]**2) for i in range(len(phi3))]
    #     tension1 = scipy.integrate.simpson(dphi3sqdz, x=z, axis=-1)
    #     tension2 = scipy.integrate.simpson(pot1, x=z, axis=-1)
    #     tension3 = scipy.integrate.simpson(pot2, x=z, axis=-1)
    #     print((1/2)*tension1 + (1/4)*tension2 + (1/2)*tension3 )
    #     return (1/2)*tension1 + (1/4)*tension2 + (1/2)*tension3 
    def GetTension(self, z, field_values, beta): 
        u = (1/np.sqrt(3+2* beta))
        phi1, phi2, phi3  = field_values[...,0]/u, field_values[...,1]/u, field_values[...,2]/u
        dphi3sqdz = [((((phi3[i+1] - phi3[i])/(z[i+1] - z[i]))**2) +(((phi2[i+1] - phi2[i])/(z[i+1] - z[i]))**2)+ (((phi1[i+1] - phi1[i])/(z[i+1] - z[i]))**2)) for i in range(len(phi3) - 1)]
        dphi3sqdz.append(0)
        pot1 = [(phi2[i]**2 + phi1[i]**2 + phi3[i]**2 -3)**2  for i in range(len(phi3))]
        pot2 = [beta*((phi1[i]**2)*(phi3[i]**2)+ (phi1[i]**2) * (phi2[i]**2) + (phi2[i]**2) * (phi3[i]**2)-3)for i in range(len(phi3))]
        #pot3 = [Arep*(phi2[i]*phi1[i]*phi3[i]-1) for i in range(len(phi3))]/(u**2)
        tension1 = scipy.integrate.simpson(dphi3sqdz, x=z, axis=-1)
        tension2 = (u**2)*scipy.integrate.simpson(pot1, x=z, axis=-1)
        tension3 = (u**2)*scipy.integrate.simpson(pot2, x=z, axis=-1)
        #tension4 = scipy.integrate.simpson(pot3, x=z, axis=-1)
        print(tension1) #+ (1/4)*tension2 + (1/2)*tension3) #+ tension4 )
        return tension1 #+ (1/4)*tension2 + (1/2)*tension3  #+ tension4

        
    def GetPartialTension(self, z, field_values, delta, beta):
        phi1, phi2, phi3  = field_values[...,0], field_values[...,1], field_values[...,2]
        dphi3sqdz = [((((phi3[i+1] - phi3[i])/(z[i+1] - z[i]))**2) +(((phi2[i+1] - phi2[i])/(z[i+1] - z[i]))**2)+ (((phi1[i+1] - phi1[i])/(z[i+1] - z[i]))**2)) for i in range(len(phi3) - 1)]
        dphi3sqdz.append(0)
        pot1 = [ (phi2[i]**2 + phi1[i]**2 + phi3[i]**2 -1)**2 for i in range(len(phi3))]
        pot2 = [ beta*(phi1[i]**2)*(phi3[i]**2) for i in range(len(phi3))]
        
        x = np.array(z)
       
        y3 = np.array(dphi3sqdz)
        y2 = np.array(pot2)
        y1 = np.array(pot1)
        # Create a boolean mask for the condition x[i] < -delta/2 or x[i] > delta/2
        mask = (x >= -delta/2) & (x <= delta/2)
    
        # Apply the mask to both x and y
        x_filtered = x[mask]
        y3_filtered = y3[mask]
        y2_filtered = y2[mask]
        y1_filtered = y1[mask]
        tension1 = scipy.integrate.simpson(y3_filtered, x=x_filtered)
        tension2 = scipy.integrate.simpson(y2_filtered, x=x_filtered)
        tension3 = scipy.integrate.simpson(y1_filtered, x=x_filtered)
        return (1/2)*tension1 + (1/2)*tension2 +(1/4)*tension3
    
    def FindDelta(self, z, field_values, beta):
        total_tension = self.GetTension(z, field_values, beta)
        target_tension = 0.6438 * total_tension
        delta = 0

        left = 0
        right = max(z) - min(z)
        epsilon = 1e-6  # Tolerance for stopping the search

        while right - left > epsilon:
            delta = (left + right) / 2
            partial_tension = self.GetPartialTension(z, field_values, delta, beta)

            if partial_tension < target_tension:
                left = delta
            else:
                right = delta

        return delta


def makePlots():
    # Thin-walled instanton
#    plt.figure()
#    ax = plt.subplot(221)
    m = Potential(mu2=1 , g1=1., g2=-0.1, A = -0.0001)
    u = 1/np.sqrt(3-0.2)
    beta = -0.1
    Arep = -0.0001
    a  = ( Arep)/(2*np.sqrt((3 + (2 * beta))))
    am = (np.sqrt(1+ a **2) -a)
    ap = (np.sqrt(1+ a **2) +a)
    field_values = u*np.array([[am, am, am],[-ap,ap,ap],[-am,-am, am ]])

#    m.plotContour()
    Y = pd.fullTunneling(field_values, m.V, m.dV)
    #print(Y)
    np.savetxt("profile1Dzeta_Z2.txt",Y.profile1D.R)
    np.savetxt("profile1Dphi_Z2.txt",Y.Phi)
    import matplotlib.pyplot as plt

    # Example data
    x_values = Y.profile1D.R
    y_values = Y.Phi

    # Create the plot
    plt.plot(x_values, y_values)

    # Add labels and title
    plt.xlabel('z')
    plt.ylabel('Phi')
    plt.show()
    #    print("GETTTTT TOTAL",  get_total_energy(Y.profile1D[0], Y.profile1D[1], m))

def makeScan_BP():
    output = []
    Arep = -0.0001
   
    beta = -0.01
    a  = (Arep)/(2*np.sqrt((3 + (2 * beta))))
    am = (np.sqrt(1+ a **2) -a)
    ap = (np.sqrt(1+ a **2) +a)
    field_values = np.array([[am,am, am],[-ap, ap, ap ],[-am, -am, am]]) #(0.47)
    #field_values = np.array([[am,am, am],[-am, -am, am]]) #(0.64)
    field_val = (1/np.sqrt(3+2* beta))*field_values
    Model = Potential(mu2= 1,g1= 1,g2= beta, A = Arep)
    sol = pd.fullTunneling(field_val, Model.V, Model.dV)
    print(Model.GetTension(sol.profile1D.R, sol.Phi, beta))
    out = np.asarray([beta,Model.GetTension(sol.profile1D.R, sol.Phi, beta)*(3+2* beta), Model.FindDelta(sol.profile1D.R, sol.Phi, beta)]) 
    output.append(out)
        
    np.savetxt("Scan_T3_A_tension_delta.txt",output)
    return output 

def makeScan_plot():
    output = []
    bb = []
    tt = []
    aa = []
    for i in range(-40,-10): 
        for j in range(-50, -20): 
            beta = -10**(i/10)
            Arep = -10**(j/10)
            print(Arep)
            a  = (Arep)/(2*np.sqrt((3 + (2 * beta))))
            am = (np.sqrt(1+ a **2) -a)
            ap = (np.sqrt(1+ a **2) +a)
            field_values = np.array([[am,am, am],[-ap,ap,ap],[-am, -am, am]]) #(0.47)
            #field_values = np.array([[am,am, am],[-am, -am, am]]) #(0.64)
            field_val = (1/np.sqrt(3+2* beta))*field_values
            Model = Potential(mu2= 1,g1= 1,g2= beta, A = Arep)
            sol = pd.fullTunneling(field_val, Model.V, Model.dV) 
            out = np.asarray([Arep, beta,Model.GetTension(sol.profile1D.R, sol.Phi, beta)*(3+2* beta), Model.FindDelta(sol.profile1D.R, sol.Phi, beta)]) 
            aa.append(-Arep)
            bb.append(-beta)
            tt.append(Model.GetTension(sol.profile1D.R, sol.Phi, beta)*(3+2* beta))
            output.append(out)
    np.savetxt("Scan_T3_A_beta_tension_new_full.txt",output)
    '''    
    [X, Y] = np.meshgrid(aa, bb) 
  
    fig, ax = plt.subplots(1, 1) 
  
    
  
    # plots filled contour plot 
    ax.contourf(X, Y, tt)     
   
    plt.show()
    '''
    return output 


if __name__ == "__main__":
    makeScan_plot()
    