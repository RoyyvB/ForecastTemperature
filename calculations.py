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

    start = time.time()

    for row in data.itertuples():

        # Calculates the energy required to heat up a room.
        data['energy'] = (mass * specific_heat * data['sup_diff']).round(decimals=2)
        
        # Fills up zero values with the next valid value.
        # data['energy'] = data['energy'].replace(to_replace=0, method='ffill')

    print(data.head(n=15))

    # Sums up J and converts it to kWh.
    JouleSum = data['energy'].sum()
    kWhSum = (JouleSum * 0.0000002778)
    
    print("\nTotal energy use is {0:,.2f} in Joules.".format(JouleSum))
    print("This is {0:,.2f} in kWh.".format(kWhSum))

    stop = time.time()
    c_time = stop - start
    
    print("This computation took " + str((c_time / 60)) + " minutes.")

    return data

# Calculation is still incorrect.
data = CalculateEnergy(data)

letssee = data.to_csv("letssee.csv")