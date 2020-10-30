import thermopy
import pandas as pd
import time

from preprocessing import read_ahu_one, MergeData, PrepareData

data = PrepareData()

def CalculateEnergy(data):
    """Calculates the energy necessary to
    attain a certain temperature."""

    # Creates two new columns: energy and sup_diff.
    # Energy will be calculated using specific heat.
    # Sup_diff calculates the difference in supply air temp.
    data['energy'] = None
    data['energy'] = pd.to_numeric(data['energy'], downcast='integer')
    data['sup_diff'] = data['sup'].diff()
    data['sup_diff'][0] = "34.6"

    # The approximate density of air (room temperature) is 1.292 per cubic meter (m3).
    # The cubic meter per hour in Cabin J14 is 439 m3/h.
    mass = (1.292 * 439) # Kilograms.
    specific_heat = 1012 # Joule.

    energy = []
    start = time.time()

    for row in data.itertuples():

        data['energy'] = (mass * specific_heat * data['sup_diff']).round(decimals=2)

    stop = time.time()
    print(data.tail())

    JouleSum = data['energy'].sum()
    kWhSum = (JouleSum * 0.0000002778)
    
    print("\nTotal energy use is {0:,.2f} in Joules.".format(JouleSum))
    print("This is {0:,.2f} in kWh.".format(kWhSum))
    print("This computation took " + str((stop - start)) + ".")

    return data

# Calculation is still incorrect.
data = CalculateEnergy(data)

