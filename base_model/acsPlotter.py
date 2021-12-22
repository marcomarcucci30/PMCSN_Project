from matplotlib import pyplot as plt
import json
lag = 11
with open("acs.json", 'r') as f:
    data = json.load(f)
    f.close()

autocorr_32 = data["32"][:lag]
autocorr_64 = data["64"][:lag]
autocorr_128 = data["128"][:lag]
autocorr_256 = data["256"][:lag]
lags = [ j for j in range (len(autocorr_128)) ][:lag]
'''
autocorr_32 = autocorr_32[:5:]
autocorr_64 = autocorr_64[:5:]
autocorr_128 = autocorr_128[:5:]
autocorr_256 = autocorr_256[:5:]
lags = lags[:5:]
'''
plt.plot( lags, autocorr_32, linestyle='--', marker='o', color='m')
plt.plot( lags, autocorr_64, linestyle='--', marker='o', color='b' )
plt.plot( lags, autocorr_128, linestyle='--', marker='o', color='r' )
plt.plot( lags, autocorr_256, linestyle='--', marker='o', color='g' )
plt.axhline(y=0.0, color='k')

plt.legend(["b = 32", "b = 64", "b = 128", "b = 256"])
plt.title("Autocorrelation (in function of lag j)")

plt.xlabel( "lag j" )
plt.ylabel( "autocorrelations r[j]" )
plt.ylim( -0.3, 1.0 )

plt.savefig("AUTOCORRELATIONS.png", dpi=350)
plt.close()


