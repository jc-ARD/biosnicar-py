# Functions to reprocess spectral and broadband surface albedo data from MOSAiC
# M. Smith, 05/2021

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from datetime import datetime
import re

def get_line_info(linenum):
    #get full line name ("linename") and shortname for file naming ("shortname") 
    #input: linenum from raw saved files
    
    if linenum == 'is':
        linename = 'Ice Station' 
        shortname = 'IS'
    if linenum == '1':
        linename = 'Lemon drop line' 
        shortname = 'LDL'
    if linenum == '2':
        linename = 'Root beer line' 
        shortname = 'RBB'
    if linenum == 'rov':
        linename = 'ROV grid'
        shortname = 'ROV'
    if linenum == 'old':
        linename = 'SYI Albedo Line'
        shortname = 'SYI'
    if linenum == 'rs':
        linename = 'Reunion Stakes'
        shortname = 'RS'
    if linenum == 'db':
        linename = 'Drone Bones'
        shortname = 'DB'
    if linenum == 'stern':
        linename = 'Stern area'
        shortname = 'STERN'
    if linenum == 'bp':
        linename = 'Bean Pole Stakes'
        shortname = 'BP'
    if linenum == 'bop':
        linename = 'Scattering layer Boptics experiment'
        shortname = 'BOP'
    if linenum == 'fyi':
        linename = 'FYI Coring area'
        shortname = 'FYI'
    if linenum == 'deck':
        linename = 'deck'
        shortname = 'deck'
    if (linenum == 'bounty') | (linenum == 'bountyline'):
        linename = 'Bounty line'
        shortname = 'BOUNTY'
    if (linenum == 'kinder') | (linenum == 'kinderline'):
        linename = 'Kinder line'
        shortname = 'KINDER'
    if (linenum == 'toblerone') | (linenum == 'tobleroneline'):
        linename = 'Toblerone line'
        shortname = 'TOBLERONE'
    if linenum == 'snowtarget':
        linename = 'Snowtarget experiment'
        shortname = 'SNOWTARGET'
    if linenum == 'bgcponds':
        linename = 'BGC Ponds'
        shortname = 'BGC'
        
    return linename, shortname


def linenum_from_shortname(shortname):
    if shortname == 'ROV3':
        linenum =''
        linename = 'ROV line, Leg 3'
    if shortname == 'SYI':
        linenum =''
        linename= 'Second-year Ice, Leg 3'
    if shortname == 'LDL':
        linenum = '1'
        linename = 'Lemon drop line'
    if shortname == 'RBB':
        linenum = '2'
        linename = 'Root beer line' 
    if shortname == 'ROV 4':
        linenum = 'rov'
        linename = 'ROV grid, Leg 4'
    if shortname == 'BOUNTY':
        linenum = 'bounty'
        linename = 'Bounty line'
    if (shortname == 'KINDER') | (shortname == 'KINDERLINE'):
        linenum = 'kinder'   
        linename = 'Kinder line'
    if shortname == 'TOBLERONE':
        linenum = 'toblerone'
        linename = 'Toblerone line'
    if shortname == 'SNOWTARGET':
        linenum = 'snowtarget'
        linename = 'Snowtarget experiment'
    if shortname == 'BGC':
        linenum = 'bgcponds'
        linename = 'BGC Ponds'
    return linenum, linename


def ship_lat_lon(datestr,timestr,LegNum):
    #get approximate latitude and longitude from ship drift track
    #inputs: date in string format, time in string format, leg number
    
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    
    dir_met = '/Users/msmith/Documents/MOSAiC/Drift/'
    df_ship = pd.read_excel(dir_met + 'PS122_' + str(LegNum) + '_link-to-mastertrack.xlsx',header=19)
    df_ship['DateTimeObj'] =  pd.to_datetime(df_ship['Date/Time'])#, format='%Y-%m-%dT%M%H'
    
    #find closest time in drifttrack to start time
    dt_closest = nearest(df_ship['DateTimeObj'],datetime.strptime('2020'+datestr+timestr,'%Y%m%d%H%M'))
    df_time = df_ship[df_ship['DateTimeObj'] == dt_closest]
    
    time_lat = float(df_time['Latitude'].values)
    time_lon = float(df_time['Longitude'].values)
    
    return time_lat,time_lon


def ASD_reprocess(filename, linename, datestr, metfile, printflag, plotflag):
    #Reprocess spectral albedos from ASD
    #inputs:
        #filename: path of .txt file to read in +reprocess
        #linename, datestr
        #metfile: path of "notes file"
    QCpath = '/Users/msmith/Documents/MOSAiC/Archiving/ASD/Files/'
    
    df = pd.read_csv(filename)
    shortfilename = filename[[m.start() for m in re.finditer('/', filename)][-1]+1:-4]
    if len(metfile) > 0:
        met = pd.read_excel(metfile,header=4)
    else:
        met = []
    df_np = df.to_numpy()
    np_size = np.shape(df_np)
    
    wavelengths = df_np[:,0]
    
    num_scan = np_size[1]-1 #total number of scans
    num_stn = int((np_size[1]-1)/2) #number of stations (scans/2)
    
    incident = df_np[:,1:num_scan:2]
    reflected = df_np[:,2:num_scan+1:2]
    albedo = reflected/incident
    albedo_filtered_INC = np.full(np.shape(albedo), np.nan)
    albedo_filtered_SLOPE = np.full(np.shape(albedo), np.nan)
    
    if False: #line in reverse
        albedo = np.fliplr(albedo)
        
    #define new data structure
    df_new = pd.DataFrame([])
    df_new['Wavelengths'] = df['Wavelength']
    
    # expressions for cutoff in variability of incident
    i_d_inc = 250 #index 250 is 600 nm
    d_inc_lim = .15 #15% change
    #exp_before = abs(incident[i_d_inc,1::]-incident[i_d_inc,0:-1])/incident[i_d_inc,1::] > d_inc_lim
    #exp_after = abs(incident[i_d_inc,1::]-incident[i_d_inc,0:-1])/incident[i_d_inc,0:-1] > d_inc_lim
    #exp_before = np.insert(exp_before, 0, False)
    #exp_after = np.append(exp_after,False)
    
    #calculate change in incident (%), average of change from before and after
    change_before = abs(incident[i_d_inc,1::]-incident[i_d_inc,0:-1])/incident[i_d_inc,1::]
    change_before = np.insert(change_before, 0, 'NaN')
    change_after = abs(incident[i_d_inc,1::]-incident[i_d_inc,0:-1])/incident[i_d_inc,0:-1]
    change_after = np.append(change_after,'NaN')
    
    change_incident = []
    for i in range(0,len(change_before)):
        change_incident.append(np.nanmean([float(change_before[i]),float(change_after[i])]).round(1)*100)
    
    for i in range(albedo.shape[1]):
        #1. Parabolic fit between 750 and 1000 as defined in ASD manual
        shift = abs(albedo[651,i]-albedo[650,i])/abs(albedo[651,i])
        if (shift) > .5:
            albedo[:,i] = np.nan
            if printflag == 1:
                print('WARNING: Sensor offset > 50% of calculated albedos! Offset = ', shift) 
           
        scale = np.ones(len(wavelengths))
        scale[400:650] = ((wavelengths[400:650] - 750)**2 / (1000-750)**2) * ((albedo[651,i]-albedo[650,i])/ (albedo[650,i]) )+ 1
        albedo[:,i] = albedo[:,i]*scale
        
        #2. Filter bad scans
        #2a. Remove scan if too much change in incident
        """if exp_before[i]| exp_after[i]:
            if printflag == 1:
                print('Scan', i, ' filtered. Incident change greater than ', str(d_inc_lim*100), '%')
            albedo_filtered_INC[:,i] = albedo[:,i]
            albedo[:,i] = np.nan
        """
        #if: 
        #2b. Remove scan if peak albedo is too low OR decreasing between 400-450 nm
        #peak_lim = 300
        slope_lim = 0.008/50
        #exp1 = df_new['Wavelengths'][np.nanargmax(albedo[0:650,i])] < peak_lim #if peak (under 1000 nm) less than 476
        exp2 = np.polyfit(df_new['Wavelengths'][0:101],albedo[0:101,i],1)[0] < -(slope_lim) #if spectral gradient 400-450 is negative
        if  exp2:# exp1 | 
            #print('Scan filtered. Peak < ', str(peak_lim), ': ' , exp1 , ', slope 400-450 < ', str(slope_lim), ': ' , exp2)
            if printflag == 1:
                print('Scan', i, ' filtered. Slope 400-450 < -', str(slope_lim), ': ' , exp2)
            albedo_filtered_SLOPE[:,i] = albedo[:,i]
            albedo[:,i] = np.nan

        #3. Filter bad parts of scans 
        alb_std = pd.Series(albedo[:,i]).rolling(20).std()
        exp_noise = alb_std > .02
        albedo[:,i][exp_noise] = np.nan
        
        #4. apply moving average to medium and long wavelengths
        albedo[850::,i] = pd.Series(albedo[:,i]).rolling(15).mean()[850::]#minor filter above 1200 nm
        albedo[1450::,i] = pd.Series(albedo[:,i]).rolling(31).mean()[1450::]#more aggressive filter above 1800 nm
      
        #5. finally, remove any spectra that still have albedos greater than 1 or less than 0 
        if any((albedo[:,i] > .99) | (albedo[:,i] < 0)):
            albedo[:,i] = np.nan
        
        df_new['Albedo' + str(i)] = albedo[:,i]
    
    #print number filtered
    num_slopefilt = sum(np.isnan(albedo_filtered_SLOPE.sum(axis=0)) == False)
    perc_slopefilt = int(round( num_slopefilt / num_stn,2) * 100)
    num_incfilt = sum(np.isnan(albedo_filtered_INC.sum(axis=0)) == False)
    perc_incfilt = int(round( num_incfilt / num_stn,2) * 100)

    if printflag == 1:
        print(str(num_slopefilt), '/', str(num_stn), ' = ', str(perc_slopefilt), '% filtered due to low-wavelength slope') 
        print(str(num_incfilt), '/', str(num_stn), ' = ', str(perc_incfilt), '% filtered due to incident changing')
    
    if plotflag == 1:

        num_stn = df_new.shape[1]-1

        colors = plt.cm.jet(np.linspace(0,1,num_stn))
        fig = plt.figure(figsize=(8,5))
        fig, ax = plt.subplots(5, 1,figsize=(8,18))

        for i in range(num_stn):
            ax[0].plot(df_new['Wavelengths'],incident[:,i], color=colors[i])
        #plt.ylim([0,1])
        ax[0].set_xlim([300,2600])
        ax[0].set_ylabel('incident')
        ax[0].set_xlabel('wavelength')
        ax[0].grid()
        if len(met) > 1 :
            if 'snow' in metfile == False:
                ax[0].legend(met['Position'],ncol=3,bbox_to_anchor=(1.05,.3))

        ax[0].set_title('ASD: Raw spectra (' + linename + ') ' + datestr[2:4] + '/' + datestr[0:2] + '/2020')

        #REFLECTED

        for i in range(num_stn):
            ax[1].plot(df_new['Wavelengths'],reflected[:,i], color=colors[i])
        ax[1].set_xlim([300,2600])
        ax[1].set_ylabel('reflected')
        ax[1].set_xlabel('wavelength')
        ax[1].grid()

        #ALBEDO - raw, all
        for i in range(num_stn):
            ax[2].plot(df_new['Wavelengths'],reflected[:,i]/incident[:,i], color=colors[i])
        ax[2].set_xlim([300,2600])
        ax[2].set_ylim([-.1,1.1])
        ax[2].set_ylabel('albedo (raw)')
        ax[2].set_xlabel('wavelength')
        ax[2].grid()

        #ALBEDO, QC'd
        for i in range(num_stn):
            ax[3].plot(df_new['Wavelengths'],albedo[:,i], color=colors[i])
        ax[3].set_xlim([300,2600])
        ax[3].set_ylim([-.1,1.1])
        ax[3].set_ylabel('albedo (QC`d)')
        ax[3].set_xlabel('wavelength')
        ax[3].grid()

        #ALBEDO - filtered
        for i in range(num_stn):
            ax[4].plot(df_new['Wavelengths'],albedo_filtered_INC[:,i], color=colors[i])
            ax[4].plot(df_new['Wavelengths'],albedo_filtered_SLOPE[:,i], color=colors[i],linestyle = '--')
        ax[4].set_xlim([300,2600])
        ax[4].set_ylim([-.1,1.1])
        ax[4].set_ylabel('albedo (removed)')
        ax[4].set_xlabel('wavelength')
        ax[4].grid()
        slope_lim = 0.008/50
        ax[4].legend(['incident changing: ' + str(perc_incfilt) + '%','slope(400-450 nm) < -' + str(slope_lim)+': '+ str(perc_slopefilt)+'%'],bbox_to_anchor=(1.05,.2))


        if True:
            QCplotpath = QCpath + 'QCplots/'
            plt.savefig(QCplotpath + shortfilename + '_rawSpectra.jpg',bbox_inches='tight',dpi=120)
        plt.close()
    
    return df_new, wavelengths, albedo, incident, reflected, num_stn, albedo_filtered_INC, albedo_filtered_SLOPE, change_incident



def ASD_df_tocsv_wheader(df, change_incident, newfilename, metfile, line,datestr, timestr, LegNum):
    #Save ASD reprocessed datafile (pandas) to csv, with header (time, lat, lon, etc.)
    #uses outputs from ASD_reprocess
    
    linename, shortname = get_line_info(line)
    
    #read in met file, get line info only, reorganize for saving to csv
    met = pd.read_excel(metfile,header=4)
    met_line = met[(met['Line'] == shortname)]
    if len(timestr) > 0:
        met_line = met_line[met_line['Start time'] == int(timestr)]
    met_line = met_line.rename(columns={'Position': 'Position', 'snow or pond?': 'Surface type', 'Total snow/pond thickness (cm)': 'Surface thickness (cm)', 'Notes on footprint': 'Notes'})

    #read in met file again to get addl header information
    met_header = pd.read_excel(metfile,header=None)
    start_time = str(met_header.iloc[0,1]) 
    sky_cond = str(met_header.iloc[2,1])
    other_notes = str(met_header.iloc[3,1])
    
    if 'Start time' in met_line.keys():
        start_time = int(met_line['Start time'].iloc[0])
        print('Using start time from column', start_time)
    
    #add ship's lat & lon to header
    time_lat, time_lon = ship_lat_lon(datestr, str(start_time), LegNum)
    
    #write a new csv with the header and data
    
    # write the header
    header = 'Start time (UTC), '+ str(start_time).zfill(4) + ' \n Sky conditions, '+ sky_cond+' \n Other notes, ' +other_notes+' \n Ship latitude, '+str(time_lat)+'\n Ship longitude, '+str(time_lon)+'\n'
    with open(newfilename, 'w') as fp:#[0:-5] + 'data.csv'
        fp.write(header)
        fp.write('Position,'+str(met_line['Position'].to_string(index=False)).replace("\n", ",")+'\n')#.astype(int)
        fp.write('Surface type,'+str(met_line['Surface type'].to_string(index=False)).replace("\n", ",")+'\n')
        fp.write('Surface thickness (cm),'+str(met_line['Surface thickness (cm)'].to_string(index=False)).replace(",", ".").replace("\n", ",")+'\n')
        fp.write('Notes,'+(met_line['Notes'].to_string(index=False).replace(",", ".").replace("\n", ","))+'\n')
        fp.write('Change in incident (%),'+str(change_incident)[1:-1]+'\n')
    df.to_csv(newfilename, header=True,index=False, mode='a', float_format='%.3f') 
    return met_line



def Kipps_albedownotes_tocsv(df_line, newfilename, metfile, line, datestr,LegNum):
    #Reprocess and resave Kipps broadband albedo data for Legs 4 and 5
    
    #read in met file, get line info only, reorganize for saving to csv
    met = pd.read_excel(metfile,header=4)
    met_line = met[(met['Line'] == line)]

    met_line_info = met_line.loc[:,['Position','snow or pond?','Total snow/pond thickness (cm)','Notes on footprint']]
    met_line_info = met_line_info.rename(columns={'Position': 'Position', 'snow or pond?': 'Surface type', 'Total snow/pond thickness (cm)': 'Surface thickness (cm)', 'Notes on footprint': 'Notes'})
    
    #merge notes and albedo df using "position" column
    df_line = df_line.rename(columns={'Position (m)': 'Position', 'Incoming': 'Incoming (microV)', 'Incoming (corrected)': 'Incoming (W m^-2)', 
                                      'Reflected': 'Reflected (microV)', 'Reflected(corrected)': 'Reflected (W m^-2)', 'Albedo (corrected)': 'Albedo'})
    df_line['Albedo'] = df_line['Albedo'].round(3)
    df_line['Incoming (W m^-2)'] = df_line['Incoming (W m^-2)'].round(3)
    df_line['Reflected (W m^-2)'] = df_line['Reflected (W m^-2)'].round(3)
    
    if line == 'SNOWTARGET':
        df_met_merged = df_line.merge(met_line_info,how='left',left_index=True, right_index=True)
        df_met_merged = df_met_merged.drop(columns=['Position_y']).rename(columns={'Position_x': 'Position'})
    else:
        df_met_merged = df_line.merge(met_line_info,how='left',on='Position')
    #print(df_met_merged)
    df_met_merged = df_met_merged[['Position','Surface type','Surface thickness (cm)','Notes','Incoming (microV)','Incoming (W m^-2)','Reflected (microV)','Reflected (W m^-2)','Albedo']]
    
    #QC flags
    #1. add estimate for error based on resolution of instrument
    k_acc = 0.01 #assumed accuracy of kipps measurement 
    df_met_merged['Albedo error'] = ((df_met_merged['Reflected (microV)']+.01)/(df_met_merged['Incoming (microV)']-.01) - (df_met_merged['Reflected (microV)']-.01)/(df_met_merged['Incoming (microV)']+.01)).round(3)
    
    #2. add percent change of incident compared to before/after
    df_met_merged['Incoming change (%)'] = ""
    incoming_diff = df_met_merged['Incoming (microV)'].diff().abs()
    for di in range(0,len(df_met_merged)-1):
        df_met_merged['Incoming change (%)'][di] = (np.nanmean([incoming_diff[di],incoming_diff[di+1]])/df_met_merged['Incoming (microV)'][di]*100).round(1)
    df_met_merged['Incoming change (%)'][len(incoming_diff)-1] = (incoming_diff[len(incoming_diff)-1]/df_met_merged['Incoming (microV)'][len(incoming_diff)-1]*100).round(1)
    
    #read in met file again to get addl header information
    met_header = pd.read_excel(metfile,header=None)
    start_time = str(met_header.iloc[0,1]) 
    sky_cond = str(met_header.iloc[2,1])
    other_notes = str(met_header.iloc[3,1])
    
    if 'Start time' in met_line.keys():
        start_time = str(int(met_line['Start time'].iloc[0])).zfill(4)
        print('Using start time from column', start_time)
    
    #add ship's lat & lon to header
    time_lat, time_lon = au.ship_lat_lon(datestr, start_time,LegNum)
    
    #write a new csv with the header and data
    # write the header
    header = 'Start time (UTC), '+ str(start_time)+ ' \n Sky conditions, '+ sky_cond+' \n Other notes, ' +other_notes+' \n Ship latitude, '+str(time_lat)+'\n Ship longitude, '+str(time_lon)+'\n'
    
    with open(newfilename, 'w') as fp:
        fp.write(header)

    # write the rest
    df_met_merged.to_csv(newfilename, header=True,index=False, mode='a')
    
    return df_met_merged
    
def Kipps_albedoLeg3_tocsv(df_line, notes_line, newfilename, datestr):
    #Reprocess and resave Kipps broadband albedo for Leg 3 (different file format)
    df_line_new = df_line[['Position', 'Snow depth [cm]','Notes','Incoming (raw)','Incoming (corrected)', 'Reflected (raw)', 'Reflected (corrected)','Albedo (corrected)']]
    df_line_new = df_line_new.rename(columns={'Snow depth [cm]': 'Surface thickness (cm)', 'Incoming (raw)': 'Incoming (microV)', 'Incoming (corrected)': 'Incoming (W m^-2)', 
                                          'Reflected (raw)': 'Reflected (microV)', 'Reflected (corrected)': 'Reflected (W m^-2)', 'Albedo (corrected)': 'Albedo'})
    df_line_new['Albedo'] = df_line_new['Albedo'].round(3)
    df_line_new['Incoming (W m^-2)'] = df_line_new['Incoming (W m^-2)'].round(3)
    df_line_new['Reflected (W m^-2)'] = df_line_new['Reflected (W m^-2)'].round(3)

    #QC flags
    #1. add estimate for error based on resolution of instrument
    k_acc = 0.01 #assumed accuracy of kipps measurement 
    df_line_new['Albedo error'] = ((df_line_new['Reflected (microV)']+.01)/(df_line_new['Incoming (microV)']-.01) - (df_line_new['Reflected (microV)']-.01)/(df_line_new['Incoming (microV)']+.01)).round(3)

    #2. add percent change of incident compared to before/after
    df_line_new['Incoming change (%)'] = ""
    incoming_diff = df_line_new['Incoming (microV)'].diff().abs()
    for di in range(0,len(df_line_new)-1):
        df_line_new['Incoming change (%)'][di] = (np.nanmean([incoming_diff[di],incoming_diff[di+1]])/df_line_new['Incoming (microV)'][di]*100).round(1)
    df_line_new['Incoming change (%)'][len(incoming_diff)-1] = (incoming_diff[len(incoming_diff)-1]/df_line_new['Incoming (microV)'][len(incoming_diff)-1]*100).round(1)

    start_time = df_line['Time (UTC)'][0].strftime('%H%M').zfill(4)
    #add ship's lat & lon to header
    time_lat, time_lon = au.ship_lat_lon(datestr, start_time,3)

    #write a new csv with the header and data
    # write the header
    header = 'Start time (UTC), '+ str(start_time)+ ' \n Sky conditions,' +str(notes_line['Weather'].iloc[0]) +  '\n Other notes,' +str(notes_line['Notes'].iloc[0]) +  '\n Ship latitude, '+str(time_lat)+'\n Ship longitude, '+str(time_lon)+'\n'

    with open(newfilename, 'w') as fp:
        fp.write(header)

    # write the rest
    df_line_new.to_csv(newfilename, header=True,index=False, mode='a')

    return df_line_new