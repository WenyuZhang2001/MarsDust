import pandas as pd
import matplotlib.pyplot as plt

# Set threshold and identify dust signals
# load Signal Data with Probability
year = 2015
month_list = range(1,13)
day_list = range(1,32)

for month in month_list:
    for day in day_list:
        try:
            SignalData = pd.read_csv(f'MAVEN_SignalProbability_model5/mvn_lpw_l2_we12burstmf_'
                                 f'{year:0>4d}{month:0>2d}{day:0>2d}_SignalProbability.csv')
        except:
            print(f'No file: MAVEN_SignalProbability_model5/'
                  f'mvn_lpw_l2_we12burstmf_{year:0>4d}{month:0>2d}{day:0>2d}_SignalProbability.csv')
            continue

        SignalData.drop(columns=['Unnamed: 0.1','Unnamed: 0'], inplace=True)
        # Check the Dust Signal data
        Probability_threshold = 0.9 # set threshold
        # print(SignalData.head())
        DustSignalData = SignalData.loc[SignalData['Probability']>=Probability_threshold]
        DustSignalData = pd.DataFrame(DustSignalData)
        print(f'DustSignal on day {year:0>4d}{month:0>2d}{day:0>2d} numbers = ',len(DustSignalData))
        DustSignalData.to_csv('MAVEN_DustImpactSignal/model5_DustSignal/mvn_lpw_l2_we12burstmf_'
                                 f'{year:0>4d}{month:0>2d}{day:0>2d}_DustImpactSignal.csv')

