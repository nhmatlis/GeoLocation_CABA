import itertools
import os
import zipfile
import datetime
from datetime import datetime
import json
import numpy as np
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.figure import figaspect
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.stats import linregress
import matplotlib
import random
from scipy.stats import ks_2samp
from numpy.random import seed
from numpy.random import randn
from numpy.random import lognormal
from scipy.stats import norm
    


def smartdir(my_dir):
    if not os.path.exists(my_dir):
        os.mkdir(my_dir) # creates directory/file
    return my_dir

def GetData(CorpusDir):
    print('Getting Data...')
    tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = [], [], [], [], [], [], [], []
    count = 0
    files = os.listdir(CorpusDir)
    print('Number of files in CorpusDir: ', len(os.listdir(CorpusDir)))
    for filename in files: #loop over files in Corpus
        starttime = datetime.now()
        print(starttime,filename)
        d=0
        try:
            with open(CorpusDir +'/'+ filename, 'r') as f:
                for line in f: # loop over lines within Corpusfile
                    try:
                        j=json.loads(line)
                        tweet_IDs.append(j["user"]["id"])
                        tweet_lons.append(j["coordinates"]["coordinates"][0])
                        tweet_lats.append(j["coordinates"]["coordinates"][1])
                        tweet_texts.append(j["text"].replace('\t',' ').replace('\n',' '))
                        tweet_dates.append(j["created_at"])
                        tweet_times.append(j["timestamp_ms"])
                        if j["user"]["name"] != None:
                            tweet_names.append(j["user"]["name"])
                        else:
                            tweet_names.append('')
                        if j["user"]["location"] != None:
                            tweet_locations.append(j["user"]["location"])
                        else:
                            tweet_locations.append('')
                        count += 1
                        if False and count%10000 == 0:
                            print('Number of JSON Objects processed: ', count)
                    except:
                        print('Problem with json. d = ', d)
        except:
            print('skipping. d = ', d)
        endtime = datetime.now()
        print('Subfile processing time: ', endtime - starttime)
    return tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations
    #tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)

def GetTweetIDs(CorpusDir):
    print('Getting Data...')
    tweet_IDs = []
    count = 0
    files = os.listdir(CorpusDir)
    print('Number of files in CorpusDir: ', len(os.listdir(CorpusDir)))
    for filename in files: #loop over files in Corpus
        starttime = datetime.now()
        print(starttime,filename)
        d=0
        try:
            with open(CorpusDir +'/'+ filename, 'r') as f:
                for line in f: # loop over lines within Corpusfile
                    try:
                        j=json.loads(line)
                        tweet_IDs.append(j["id"])
                        count += 1
                        if False and count%10000 == 0:
                            print('Number of JSON Objects processed: ', count)
                    except:
                        print('Problem with json. d = ', d)
        except:
            print('skipping. d = ', d)
        endtime = datetime.now()
        print('Subfile processing time: ', endtime - starttime)
    return tweet_IDs
    #tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)

def GetRandomInds(NumSamples,NumSelection):
	RInds = [int(random.uniform(0,NumSamples)) for i in range(int(1.1*NumSelection))]
	RInds_U = sorted(list(set(RInds))[0:NumSelection])
	return RInds_U

def BinUsers(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations):
    starttime = datetime.now()
    print('Binning Users...')
    IDs = sorted(list(set(tweet_IDs)))
    ID_counts = [0 for x in IDs]
    ID_lons, ID_lats, ID_texts, ID_dates, ID_times = [[] for x in IDs], [[] for x in IDs], [[] for x in IDs], [[] for x in IDs], [[] for x in IDs]
    ID_names, ID_locations = [[] for x in IDs], [[] for x in IDs]
    count = 0
    for i in range(len(tweet_IDs)):
        idx = IDs.index(tweet_IDs[i]) #find the index
        ID_counts[idx] += 1
        ID_lons[idx].append(tweet_lons[i]) # add element to Longitude list
        ID_lats[idx].append(tweet_lats[i]) # add element to Latitude list
        ID_texts[idx].append(tweet_texts[i])
        ID_dates[idx].append(tweet_dates[i])
        ID_times[idx].append(tweet_times[i])
        ID_names[idx].append(tweet_names[i])
        ID_locations[idx].append(tweet_locations[i])
        count += 1
        if count%10000 == 0:
            print('Number of Tweets processed: ', count)
    endtime = datetime.now()
    print('Loading processing time: ', endtime - starttime)
    return IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations

def QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations):
    starttime = datetime.now()
    print('Binning data by user ID.')
    print('Data format: tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations')
    Sorted_Tweets = sorted(zip(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations))
    SortedElts = []
    for x in Sorted_Tweets:
        SortedElts.append(list(x))
    Sorted_IDs = sorted(tweet_IDs)
    ChInds = [i for i in range(1,len(Sorted_IDs)) if Sorted_IDs[i] != Sorted_IDs[i-1]]
    LimInds = [0]+ChInds+[len(SortedElts)]
    BinnedElts = [SortedElts[LimInds[i]:LimInds[i+1]] for i in range(len(LimInds)-1)]
    ID_User_Data = []
    ID_User_Data.append([x[0][0] for x in BinnedElts])
    ID_User_Data.append([len(x) for x in BinnedElts])
    for i in range(1,8):
        ID_User_Data.append([[y[i] for y in x] for x in BinnedElts])
    endtime = datetime.now()
    print('Binning processing time: ', endtime - starttime)
    return ID_User_Data, BinnedElts
    #ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)

def QuickBinCoord(lons, lats):
    starttime = datetime.now()
    print('Binning coordinates...')
    ComplexCoord = [complex(lons[i],lats[i]) for i in range(len(lons))]
    Sorted_CCoord = sorted(ComplexCoord)
    ChInds = [i for i in range(1,len(Sorted_CCoord)) if Sorted_CCoord[i] != Sorted_CCoord[i-1]]
    LimInds = [0]+ChInds+[len(Sorted_CCoord)]
    Freq = [LimInds[i+1] - LimInds[i] for i in range(len(LimInds)-1)]
    BinnedCoord = [Sorted_CCoord[LimInds[i]] for i in range(len(LimInds)-1)]
    BinnedLons = [x.real for x in BinnedCoord]
    BinnedLats = [x.imag for x in BinnedCoord]
    endtime = datetime.now()
    print('Binning processing time: ', endtime - starttime)
    return BinnedLons, BinnedLats, Freq
    #BinnedLons, BinnedLats, Freq = QuickBinCoord(lons, lats)

def CheckUserDatumOK(datum):
    return all([datum[1] == len(datum[i]) for i in range(2,9)])

def CheckUserDataOK(ID_User_Data):
    NumUsers = len(ID_User_Data[0])
    Status = []
    BadUserElts = []
    for i in range(NumUsers):
        datum = [x[i] for x in ID_User_Data]
        State = CheckUserDatumOK(datum)
        Status.append(State)
        if not State:
            BadUserElts.append(datum)
        if i%10000 == 0:
            print(i)
    DataState = all(Status)
    print('Number of users checked: {}. Data is OK: {}. Number of bad users: {}'.format(NumUsers,OverallState, len(BadUserElts)))
    return DataState

def SaveBinnedData(SavePath, IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations):
    print('Saving user-binned data to file...')
    with open(SavePath, 'w', encoding='utf-8') as ff:
        print('User ID, Number of Tweets, longitude list, latitude list, text list, date list, ms timer list', file=ff) #Create header
        for i in range(len(IDs)):
            print(IDs[i], ID_counts[i], ID_lons[i], ID_lats[i], ID_texts[i], ID_dates[i], ID_times[i], ID_names[i], ID_locations[i], sep = '\t', file=ff)
            if i%10000 == 0:
                print('Number of Tweets saved: ', i)

def SaveMatrixData(SavePath, lons, lats, counts_ref, counts_word):
    print('Saving user-binned data to file...')
    with open(SavePath, 'w', encoding='utf-8') as ff:
        print('\t'.join(['Longitudes', 'Latitudes', 'Ref Counts', 'Word Counts']), file=ff) #Create header
        for i in range(len(lons)):
            print('{:6f}'.format(lons[i]), '{:6f}'.format(lats[i]), counts_ref[i], counts_word[i], sep = '\t', file=ff)
            if i%10000 == 0:
                print('Number of Tweets saved: ', i)
    #SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)

def LoadMatrixData(DataPath):
    print('Loading matrix data from file...')
    with open(DataPath, 'r', encoding='utf-8') as ff:
        ListLines, ListBadLines =[], []
        header = ff.readline()
        lons, lats, counts_ref, counts_word = [],[],[],[]
        count, count_err = 0, 0
        for line in ff:
            try:
                ListLines.append(line)
                count += 1
                LineElts = line.strip().split('\t')
                lons.append(float(LineElts[0]))
                lats.append(float(LineElts[1]))
                counts_ref.append(int(LineElts[2]))
                counts_word.append(int(LineElts[3]))
                if count%10000 == 0:
                    print('Lines Loaded: {}'.format(count))
            except:
                ListBadLines.append(line)
                count_err += 1
                if count_err%100 == 0:
                    print('Line parsing failed. Number of failed lines: {}'.format(count_err))
    Data = [lons,lats,counts_ref,counts_word]
    print(header.strip())
    print('{}\t{}\t{}\t\t{}'.format(lons[0],lats[0],counts_ref[0],counts_word[0]))
    print('{}\t{}\t{}\t\t{}'.format(lons[1],lats[1],counts_ref[1],counts_word[1]))
    return Data #, ListLines, ListBadLines
    #Data, ListLines, ListBadLines = LoadMatrixData(DataPath)

def MoranIData(DataPath):
    print('Loading Moran\'s I data from file...')
    with open(DataPath, 'r', encoding='utf-8') as ff:
        ListLines, ListBadLines =[], []
        header = ff.readline()
        NumLayers, MoranI = [],[]
        count, count_err = 0, 0
        for line in ff:
            try:
                ListLines.append(line)
                count += 1
                LineElts = line.strip().split('\t')
                NumLayers.append(float(LineElts[0]))
                MoranI.append(float(LineElts[1]))
                if count%10000 == 0:
                    print('Lines Loaded: {}'.format(count))
            except:
                ListBadLines.append(line)
                count_err += 1
                if count_err%100 == 0:
                    print('Line parsing failed. Number of failed lines: {}'.format(count_err))
    Data = [NumLayers, MoranI]
    print(header.strip())
    print('{}\t{}'.format(NumLayers[0],MoranI[0]))
    print('{}\t{}'.format(NumLayers[1],MoranI[1]))
    return Data #, ListLines, ListBadLines


def ParseLine(line):
    Elts = line.split('\t')
    Elts[0] = int(Elts[0])
    Elts[1] = int(Elts[1])
    Elts[2] = json.loads(Elts[2])
    Elts[3] = json.loads(Elts[3])
    texts = Elts[4].split("\', \'")
    texts[0] = texts[0][2:-1]
    Elts[4] = texts
    Elts[6] = json.loads(Elts[6].replace("'",""))
    Elts[7] = json.loads(Elts[7])
    Elts[8] = json.loads(Elts[8])
    return Elts[0:9]

def GetListElts(ListLines):
    print('Parsing lines containing Tweet data into elements...')
    print('Number of lines of data: {}'.format(len(ListLines)))
    List_User_Elts, ListUnparsedLines = [], []
    count, count_err = 0, 0
    for line in ListLines:
        try:
            ID_User_Data.append(ParseLine(line))
            count += 1
            if count%10000 == 0:
                print('Lines Processed: {}'.format(count))
        except:
            ListUnparsedLines.append(line)
            count_err += 1
            if count_err%100 == 0:
                print('Number of bad lines: {}'.format(count_err))
    return List_User_Elts, ListUnparsedLines
    #List_User_Elts, ListUnparsedLines = GetListElts(ListLines)

def LoadBinnedData(DataPath):
    print('Loading user-binned Tweets from file...')
    with open(DataPath, 'r', encoding='utf-8') as ff:
        ListLines, ListBadLines =[], []        
        header = ff.readline()
        count, count_err = 0, 0
        for line in ff:
            try:
                ListLines.append(line)
                count += 1
                if count%10000 == 0:
                    print('Lines Loaded: {}'.format(count))
            except:
                ListBadLines.append(line)
                count_err += 1
                if count_err%100 == 0:
                    print('Tweet loading failed. Number of failed Tweets: {}'.format(count_err))
    List_User_Elts, ListUnparsedLines = GetListElts(ListLines)
    ListBadLines = ListBadLines + ListUnparsedLines
    return List_User_Elts, ListBadLines
    #List_User_Elts, ListBadLines = LoadBinnedData(DataPath)

def LoadTabFile(Path):
	Lines,BadLines = LoadBinnedData(Path)
	RowData = [x.strip().split('\t') for x in BadLines]
	NumLines = len(RowData)
	NumElts = len(RowData[0])
	ColData = [[x[i] for x in RowData] for i in range(NumElts)]
	return ColData

def LoadTSVFile(Path):
    print('Loading tab-separated values from file...')
    with open(Path, 'r', encoding='utf-8') as ff:
        ListLines = []
        header = ff.readline()
        for line in ff:
            ListLines.append(line)
        RowData = [line.strip().split('\t') for line in ListLines]
        NumLines = len(RowData)
        NumElts = len(RowData[0])
        ColData = [[x[i] for x in RowData] for i in range(NumElts)]
    return ColData

def LoadSpreadsheetFileAsText(Path, Delimiter):
    print('Loading spreadsheet values from file...')
    with open(Path, 'r', encoding='utf-8') as ff:
        ListLines = []
        header = ff.readline()
        for line in ff:
            ListLines.append(line)
        RowData = [line.strip().split(Delimiter) for line in ListLines]
        NumLines = len(RowData)
        NumElts = len(RowData[0])
        ColData = [[x[i] for x in RowData] for i in range(NumElts)]
    return ColData

def LoadTSV_MixedLineFile(Path):
	print('Loading tab-separated values from file...')
	with open(Path, 'r', encoding='utf-8') as ff:
		ListLines = []
		header = ff.readline()
		for line in ff:
			ListLines.append(line)
		RowData = [line.strip().split('\t') for line in ListLines]
		NumEltsList = [len(x) for x in RowData]
		NumElts = max(NumEltsList)
		UniformLists = max(NumEltsList) == min(NumEltsList)
		if UniformLists:
			ColData = [[x[i] for x in RowData] for i in range(NumElts)]
		else:
			ColData = [[] for i in range(NumElts)]
			for datum in RowData:
				for i in range(NumElts):
					if i < len(datum):
						ColData[i].append(datum[i])
					else:
						ColData[i].append('')
	return ColData, ListLines


def GetUserInds(ID_User_Data,UserIDList):
    starttime = datetime.now()
    print('Selecting users from list of user IDs..')
    #Find indices of users with specific user IDs
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_User_Inds = [] # List of indices associated with users satisfying User-Type condition
    IDs = ID_User_Data[0]
    Ind = 0
    for ID in IDs:
        if ID in UserIDList:
            ID_User_Inds.append(Ind)
        Ind += 1
        if True and Ind%10000 == 0:
            print('Processing User: {}'.format(Ind))
    endtime = datetime.now()
    print('{} users processed. Execution time: {}'.format(Ind,endtime - starttime))
    return ID_User_Inds
    #ID_User_Inds = GetUserInds(ID_User_Data,UserIDList)

def GetUserTypeInds(ID_User_Data,UserType):
    print('Selecting users of type: ', UserType)
    #Find indices of users of a given type
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_UserType_Inds = [] # List of indices associated with users satisfying User-Type condition
    C_UserType = {}
    C_UserType['All'] = True
    ID_counts = ID_User_Data[1]
    ID_lons = ID_User_Data[2]
    ID_lats = ID_User_Data[3]
    if UserType in ['NonStatic', 'Static', 'Mover']:
        for i in range(len(ID_counts)):
            D_ID_lons = max(ID_lons[i]) - min(ID_lons[i])
            D_ID_lats = max(ID_lats[i]) - min(ID_lats[i])
            ######## Search Condition Options
            C_UserType['NonStatic'] = (D_ID_lons > 0 and D_ID_lats > 0) or ID_counts[i] == 1 #Users who move at least once
            C_UserType['Static'] = D_ID_lons == 0 and D_ID_lats == 0 and ID_counts[i] > 1 #Users who don't move at all
            C_UserType['Mover'] = len(list(set(ID_lons[i]))) == len(ID_lons[i]) and len(list(set(ID_lats[i]))) == len(ID_lats[i]) #Users who move every Tweet
            ######## End of Search Conditions
            if C_UserType[UserType]:
                ID_UserType_Inds.append(i)
            if False and i%10000 == 0:
                print('Processing User :', i)
    elif UserType == 'All':
        ID_UserType_Inds = [i for i in range(len(ID_counts))]
    else:
        ID_UserType_Inds = []
        print('Unrecogized User Type!')
    return ID_UserType_Inds
    #ID_UserType_Inds = GetUserTypeInds(ID_User_Data,UserType)

def GetTokenUserInds(ID_User_Data, Tokens):
    print('Finding User indices filtered by tokens...')
    print('Total Number of users: {}.'.format(len(ID_User_Data[0])))
    #Get Ref and Word User Indices
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_TokenUser_Inds = [] # List of indices associated with users using a set of tokens
    ID_texts = ID_User_Data[4] 
    for Ind in range(len(ID_texts)):
        for text in ID_texts[Ind]:
            if any(wrd in text for wrd in Tokens):
                ID_TokenUser_Inds.append(Ind)
        if True and Ind%100000 == 0:
            print('Users Processed: ', Ind)
    ID_TokenUser_Inds = list(set(ID_TokenUser_Inds))
    return ID_TokenUser_Inds
    #ID_TokenUser_Inds = GetTokenUserInds(ID_User_Data, Tokens)

def GetLocationUserInds(ID_User_Data, Locations):
    print('Finding User indices filtered by locations...')
    print('Total Number of users: {}.'.format(len(ID_User_Data[0])))
    #Get Ref and Word User Indices
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_LocationUser_Inds = [] # List of indices associated with users using a set of tokens
    ID_locations = ID_User_Data[8] 
    for Ind in range(len(ID_locations)):
        for UserLocation in ID_locations[Ind]:
            if UserLocation in Locations:
                ID_LocationUser_Inds.append(Ind)
        if True and Ind%10000 == 0:
            print('Users Processed: ', Ind)
    ID_LocationUser_Inds = list(set(ID_LocationUser_Inds))
    return ID_LocationUser_Inds
    #ID_LocationUser_Inds = GetLocationUserInds(ID_User_Data, Locations)

def getYMDnum(date):
	Y = int(date.split(' ')[5])
	months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	M = 1+months.index(date.split(' ')[1])
	D = int(date.split(' ')[2])
	return Y, M, D

def getYMD_singlenum(date):
	Y = int(date.split(' ')[5])
	months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	M = 1+months.index(date.split(' ')[1])
	D = int(date.split(' ')[2])
	return Y*10000 + M*100 + D

def GetTokens(name):
    Tokens = {}
    Tokens['GramLex'] = {}
    Tokens['GramLex']['words2check'] = ["labur", "en cana", "Boludo", "boludo", " pibe", " piba", " pibit" "Labur", "tenes " , "preferís " , "abrís ", "descubrís ",  "morís " , "seguís " ,"rrís " , "erís " , "dís " , "ivís ", "guís ", "ribís ",  "tís ",  "querés ","tenés ", " sos ", " andá", " mandá", " ecís"]
    Tokens['GramLex']['RefWords'] = ["trabaj", "sigue ","sigues ", "sufres ", "abres ", "ieres ", "ives ", "dices ", "escribes ","dices ", "quieres ","tienes "," eres ", " mánda", "dime", "muy", " chic", "en prisión" ]
    Tokens['Unambiguous'] = {}
    Tokens['Unambiguous']['words2check'] = [' sos ', ' sentís ', ' sentis ', ' decís ', ' decis ', ' seguís ', ' seguis ', ' venís ', 'venis', ' vivís ', ' vivis ', ' salís ',' salis ', ' elegís ', ' elegis ', ' pedís ',' pedis ', ' referís ', ' referis ', ' preferís ', ' preferis ', ' escribís ', ' escribis ', ' subís ', ' subis ', ' recibís ', ' recibis ', ' conseguís ', ' conseguis ',  ' abrís ', ' abris ', ' sufrís ', ' sufris ', ' decidís ', 'decidis',  ' morís ', ' compartís ', 'compartis', ' definís ', 'definis', ' dormís ', ' dormis ',' coincidís ', ' coincidis ', ' convertís ', ' convertis ',' servís ', ' servis ', ' arrepentís ', ' arrepentis ', ' reís ', ' reis ', ' descubrís ', ' descubris ', ' consumís ', ' consumis ', ' existís ', 'existis ',' permitís ', ' permitis ', ' competís ', ' cometis ', ' dirigís ', ' percibís ', ' atribuís ', ' adherís ', ' vestís ', ' repetís ', ' medís ', ' asumís ', ' construís ', ' convivís ', ' divertís ', ' invertís ', ' producís ', ' describís ', ' sonreís ', ' transmitís ', ' aburrís ', ' conducís ', ' insistís ', ' rendís ', ' mentís ', ' cubrís ', ' adquirís ', ' incluís ', ' sugerís ', ' suscribís ', ' confundís ', ' exigís ', ' advertís ', ' unís ', ' recurrís ', ' dividís ', ' resistís ', ' reunís ', ' combatís ', ' emitís ', ' discutís ', ' fundís ', ' oís ', ' concebís ', ' reducís ', ' contribuís ', ' hundís ', ' inscribís ', ' nutrís ', ' prohibís ', ' sobrevivís ', ' perseguís ',' deprimís ', ' despedís ', ' aplaudís ', ' introducís ', ' corregís ', ' admitís ', ' hervís ', ' asistís ', ' aludís ', ' desmentís ', ' destruís ', ' ingerís ', ' sumergís ', ' repartís ', ' traducís ', ' transferís ', ' teñís ', ' escupís ', ' intuís ', ' acudís ', ' distribuís ', ' disminuís ', ' contradecís ', ' intervenís ', ' pulís ', ' fingís ', ' residís ', ' evadís ', ' difundís ', ' añadís ', ' curtís ', ' interrumpís ', ' revivís ', ' sacudís ', ' reprimís ', ' requerís ', ' exhibís ', ' concluís ', ' exprimís ', ' imprimís ', ' infringís ', ' herís ', ' huís ', ' presidís ', ' omitís ', ' reproducís ', ' remitís ', ' sustituís ', ' restringís ', ' resumís ', ' seducís ', ' sobresalís ', ' prevenís ', ' impartís ', ' incurrís ', ' influís ', ' concurrís ', ' comprimís ', ' aducís ', ' derretís ', ' eludís ', ' freís ', ' expandís ', ' excluís ', ' escurrís ', ' esgrimís ', ' redescubrís ', ' prostituís ', ' provenís ', ' prescindís ', ' oprimís ', ' irrumpís ', ' maldecís ', ' revertís ', ' rescindís ', ' resentís ', ' transcribís ', ' transgredís ', ' ungís ', ' zambullís ', ' reincidís ', ' reñís ', ' suprimís ', ' sucumbís ', ' invadís ', ' predecís ', ' presumís ', ' proferís ', ' esparcís ', ' esculpís ', ' eximís ', ' impedís ', ' digerís ', ' desvivís ', ' devenís ', ' desdecís ', ' desistís ', ' deducís ', ' asentís ', ' afligís ', ' adscribís ', ' ceñís ', ' aturdís ', ' consentís ', ' constituís ', ' reís ', ' subís ', ' tenés ', ' podés ', ' querés ', ' entendés ', ' mantenés ', ' pensás ', ' empezás ',' mantenés ']
    Tokens['Unambiguous']['RefWords'] = [' eres ', ' sientes ', ' dices ', ' sigues ', ' vienes ', ' vives ', ' sales ', ' eliges ', ' pides ', ' refieres ', ' prefieres ', ' escribes ', ' subes ', ' recibes ', ' consigues ', ' abres ', ' sufres ', ' decides ', ' mueres ', ' compartes ', ' defines ', ' duermes ', ' coincides ', ' conviertes ', ' sirves ', ' arrepientes ', ' ríes ', ' descubres ', ' consumes ', ' existes ', ' permites ', ' compites ', ' diriges ', ' percibes ', ' atribues ', ' adhieres ', ' vistes ', ' repites ', ' mides ', ' asumes ', ' construes ', ' convives ', ' diviertes ', ' inviertes ', ' produces ', ' descubres ', ' sonríes ', ' transmites ', ' aburres ', ' conduces ', ' insistes ', ' rindes ', ' mientes ', ' cubres ', ' adquieres ', ' inclues ', ' sugieres ', ' suscribes ', ' confundes ', ' exiges ', ' adviertes ', ' unes ', ' recurres ', ' divides ', ' resistes ', ' reunes ', ' combates ', ' emites ', ' discutes ', ' fundes ', ' oyes ', ' concibes ', ' reduces ', ' contribuyes ', ' hundes ', ' inscribes ', ' nutres ', ' prohibes ', ' sobrevives ', ' persigues ', ' deprimes ', ' despides ', ' aplaudes ', ' introduces ', ' corrigues ', ' admites ', ' hierves ', ' asistes ', ' aludes ', ' desmientes ', ' destruyes ', ' ingieres ', ' sumerges ', ' repartes ', ' traduces ', ' transfieres ', ' tiñes ', ' escupes ', ' intuyes ', ' acudes ', ' distribues ', ' disminues ', ' contradices ', ' intervienes ', ' pules ', ' finges ', ' resides ', ' evades ', ' difundes ', ' añades ', ' curtes ', ' interrumpes ', ' revives ', ' sacudes ', ' reprimes ', ' requieres ', ' exhibes ', ' concluyes ', ' exprimes ', ' imprimes ', ' infringes ', ' hieres ', ' huyes ', ' presides ', ' omites ', ' reproduces ', ' remites ', ' sustituyes ', ' restringes ', ' resumes ', ' seduces ', ' sobresales ', ' previenes ', ' impartes ', ' incurres ', ' influyes ', ' concurres ', ' comprimes ', ' aduces ', ' derrites ', ' eludes ', ' fríes ', ' expandes ', ' excludes ', ' escurres ', ' esgrimes ', ' redescubres ', ' prostituyes ', ' provienes ', ' prescindes ', ' oprimes ', ' irrumpes ', ' maldices ', ' reviertes ', ' rescindes ', ' resientes ', ' transcribes ', ' unges ', ' zambulles ', ' reincides ', ' reñes ', ' suprimes ', ' sucumbes ', ' invades ', ' predices ', ' presumes ', ' profieres ', ' esparces ', ' esculpes ', ' eximes ', ' impides ', ' digieres ', ' desvives ', ' devienes ', ' desdices ', ' desistes ', ' deduces ', ' asientes ', ' afliges ', ' adscribes ', ' aturdes ', ' consientes ', ' constituyes ', ' ríes ', ' subes ', ' tienes ', ' puedes ', ' quieres ',  ' entiendes ', ' piensas ',  ' empiezas ', ' mantienes ']
    Tokens['Unambiguous']['Description'] = 'Unambiguous verb endings. The ArgSp/words2check list contains tokens with and without accents to account for people not using them.'
    Tokens['MaxList'] = {}
    Tokens['MaxList']['words2check'] = ["Seguínos ", "Vení ", "Che,", " che," "Chau", " chau", "Bolud", "boludo", "boluda", " vos ", "Vos ", ' sos ', ' sentís ', ' sentis ', ' decís ', ' decis ', ' seguís ', ' seguis ', ' venís ', 'venis', ' vivís ', ' vivis ', ' salís ',' salis ', ' elegís ', ' elegis ', ' pedís ',' pedis ', ' referís ', ' referis ', ' preferís ', ' preferis ', ' escribís ', ' escribis ', ' subís ', ' subis ', ' recibís ', ' recibis ', ' conseguís ', ' conseguis ',  ' abrís ', ' abris ', ' sufrís ', ' sufris ', ' decidís ', 'decidis',  ' morís ', ' compartís ', 'compartis', ' definís ', 'definis', ' dormís ', ' dormis ',' coincidís ', ' coincidis ', ' convertís ', ' convertis ',' servís ', ' servis ', ' arrepentís ', ' arrepentis ', ' reís ', ' reis ', ' descubrís ', ' descubris ', ' consumís ', ' consumis ', ' existís ', 'existis ',' permitís ', ' permitis ', ' competís ', ' cometis ', ' dirigís ', ' percibís ', ' atribuís ', ' adherís ', ' vestís ', ' repetís ', ' medís ', ' asumís ', ' construís ', ' convivís ', ' divertís ', ' invertís ', ' producís ', ' describís ', ' sonreís ', ' transmitís ', ' aburrís ', ' conducís ', ' insistís ', ' rendís ', ' mentís ', ' cubrís ', ' adquirís ', ' incluís ', ' sugerís ', ' suscribís ', ' confundís ', ' exigís ', ' advertís ', ' unís ', ' recurrís ', ' dividís ', ' resistís ', ' reunís ', ' combatís ', ' emitís ', ' discutís ', ' fundís ', ' oís ', ' concebís ', ' reducís ', ' contribuís ', ' hundís ', ' inscribís ', ' nutrís ', ' prohibís ', ' sobrevivís ', ' perseguís ',' deprimís ', ' despedís ', ' aplaudís ', ' introducís ', ' corregís ', ' admitís ', ' hervís ', ' asistís ', ' aludís ', ' desmentís ', ' destruís ', ' ingerís ', ' sumergís ', ' repartís ', ' traducís ', ' transferís ', ' teñís ', ' escupís ', ' intuís ', ' acudís ', ' distribuís ', ' disminuís ', ' contradecís ', ' intervenís ', ' pulís ', ' fingís ', ' residís ', ' evadís ', ' difundís ', ' añadís ', ' curtís ', ' interrumpís ', ' revivís ', ' sacudís ', ' reprimís ', ' requerís ', ' exhibís ', ' concluís ', ' exprimís ', ' imprimís ', ' infringís ', ' herís ', ' huís ', ' presidís ', ' omitís ', ' reproducís ', ' remitís ', ' sustituís ', ' restringís ', ' resumís ', ' seducís ', ' sobresalís ', ' prevenís ', ' impartís ', ' incurrís ', ' influís ', ' concurrís ', ' comprimís ', ' aducís ', ' derretís ', ' eludís ', ' freís ', ' expandís ', ' excluís ', ' escurrís ', ' esgrimís ', ' redescubrís ', ' prostituís ', ' provenís ', ' prescindís ', ' oprimís ', ' irrumpís ', ' maldecís ', ' revertís ', ' rescindís ', ' resentís ', ' transcribís ', ' transgredís ', ' ungís ', ' zambullís ', ' reincidís ', ' reñís ', ' suprimís ', ' sucumbís ', ' invadís ', ' predecís ', ' presumís ', ' proferís ', ' esparcís ', ' esculpís ', ' eximís ', ' impedís ', ' digerís ', ' desvivís ', ' devenís ', ' desdecís ', ' desistís ', ' deducís ', ' asentís ', ' afligís ', ' adscribís ', ' ceñís ', ' aturdís ', ' consentís ', ' constituís ', ' reís ', ' subís ', ' tenés ', ' podés ', ' querés ', ' entendés ', ' mantenés ', ' pensás ', ' empezás ',' mantenés ', ' andá ']
    Tokens['MaxList']['RefWords'] = ["Sigue ", "Ven ", "asta luego", "Oye", " tú ", "Tú ", ' eres ', ' sientes ', ' dices ', ' sigues ', ' vienes ', ' vives ', ' sales ', ' eliges ', ' pides ', ' refieres ', ' prefieres ', ' escribes ', ' subes ', ' recibes ', ' consigues ', ' abres ', ' sufres ', ' decides ', ' mueres ', ' compartes ', ' defines ', ' duermes ', ' coincides ', ' conviertes ', ' sirves ', ' arrepientes ', ' ríes ', ' descubres ', ' consumes ', ' existes ', ' permites ', ' compites ', ' diriges ', ' percibes ', ' atribues ', ' adhieres ', ' vistes ', ' repites ', ' mides ', ' asumes ', ' construes ', ' convives ', ' diviertes ', ' inviertes ', ' produces ', ' descubres ', ' sonríes ', ' transmites ', ' aburres ', ' conduces ', ' insistes ', ' rindes ', ' mientes ', ' cubres ', ' adquieres ', ' inclues ', ' sugieres ', ' suscribes ', ' confundes ', ' exiges ', ' adviertes ', ' unes ', ' recurres ', ' divides ', ' resistes ', ' reunes ', ' combates ', ' emites ', ' discutes ', ' fundes ', ' oyes ', ' concibes ', ' reduces ', ' contribuyes ', ' hundes ', ' inscribes ', ' nutres ', ' prohibes ', ' sobrevives ', ' persigues ', ' deprimes ', ' despides ', ' aplaudes ', ' introduces ', ' corrigues ', ' admites ', ' hierves ', ' asistes ', ' aludes ', ' desmientes ', ' destruyes ', ' ingieres ', ' sumerges ', ' repartes ', ' traduces ', ' transfieres ', ' tiñes ', ' escupes ', ' intuyes ', ' acudes ', ' distribues ', ' disminues ', ' contradices ', ' intervienes ', ' pules ', ' finges ', ' resides ', ' evades ', ' difundes ', ' añades ', ' curtes ', ' interrumpes ', ' revives ', ' sacudes ', ' reprimes ', ' requieres ', ' exhibes ', ' concluyes ', ' exprimes ', ' imprimes ', ' infringes ', ' hieres ', ' huyes ', ' presides ', ' omites ', ' reproduces ', ' remites ', ' sustituyes ', ' restringes ', ' resumes ', ' seduces ', ' sobresales ', ' previenes ', ' impartes ', ' incurres ', ' influyes ', ' concurres ', ' comprimes ', ' aduces ', ' derrites ', ' eludes ', ' fríes ', ' expandes ', ' excludes ', ' escurres ', ' esgrimes ', ' redescubres ', ' prostituyes ', ' provienes ', ' prescindes ', ' oprimes ', ' irrumpes ', ' maldices ', ' reviertes ', ' rescindes ', ' resientes ', ' transcribes ', ' unges ', ' zambulles ', ' reincides ', ' reñes ', ' suprimes ', ' sucumbes ', ' invades ', ' predices ', ' presumes ', ' profieres ', ' esparces ', ' esculpes ', ' eximes ', ' impides ', ' digieres ', ' desvives ', ' devienes ', ' desdices ', ' desistes ', ' deduces ', ' asientes ', ' afliges ', ' adscribes ', ' aturdes ', ' consientes ', ' constituyes ', ' ríes ', ' subes ', ' tienes ', ' puedes ', ' quieres ',  ' entiendes ', ' piensas ',  ' empiezas ', ' mantienes ']
    Tokens['MaxList']['Description'] = 'Maximal list of unambiguous and tokens distinguishing ArgSp and PenSp dialects, including lexical and grammatical and balanced between ref and word sets.'
    Tokens['MaxList']['Comment'] = 'The tokens "Mira " and "Mirá " were removed from the token lists on 2021.10.05 at 19:47. The reason for removal is that the only difference between the tokens is the accent. ArgSp Tweets neglecting the accent will then be falsely identified as PenSp.'
    Tokens['ArgExtensive'] = {}
    Tokens['ArgExtensive']['words2check'] = [' re ', ' labur ', ' encana ', ' pibe ', ' piba ', ' pibit ', ' Chau ', ' chau ', ' Bolud ', ' birra ', ' bella ', ' bello ', ' yira ', ' encan ', ' encanut ', ' boch ', ' bacán ', ' bagallo ', ' bagayo ', ' bagascia ', ' balurdo ', ' berreta ', ' brodo ', ' capo ', ' cazzo ', ' crepar ', ' manyar ', ' minga ', ' nono ', ' nona ', ' parlar ', ' parla ', ' salute ', ' testa ', ' yira ', ' bolud ', ' Che, ', ' Che! ', ' pelotudo ', ' re ', ' vos ', ' cheta ', ' cheto ', ' sentís ', ' decís ', ' seguís ', ' venís ', ' vivís ', ' salís ', ' elegís ', ' pedís ', ' referís ', ' preferís ', ' escribís ', ' subís ', ' recibís ', ' conseguís ', ' abrís ', ' sufrís ', ' decidís ', ' morís ', ' compartís ', ' definís ', ' dormís ', ' coincidís ', ' convertís ', ' cumplís ', ' servís ', ' arrepentís ', ' reís ', ' descubrís ', ' consumís ', ' existís ', ' permitís ', ' competís ', ' dirigís ', ' percibís ', ' atribuís ', ' adherís ', ' vestís ', ' repetís ', ' medís ', ' asumís ', ' construís ', ' convivís ', ' divertís ', ' invertís ', ' producís ', ' describís ', ' sonreís ', ' transmitís ', ' aburrís ', ' conducís ', ' insistís ', ' rendís ', ' mentís ', ' cubrís ', ' adquirís ', ' partís ', ' incluís ', ' sugerís ', ' suscribís ', ' confundís ', ' exigís ', ' advertís ', ' unís ', ' recurrís ', ' dividís ', ' resistís ', ' reunís ', ' combatís ', ' emitís ', ' discutís ', ' fundís ', ' oís ', ' concebís ', ' reducís ', ' contribuís ', ' hundís ', ' inscribís ', ' nutrís ', ' prohibís ', ' sobrevivís ', ' perseguís ', ' lucís ', ' deprimís ', ' despedís ', ' aplaudís ', ' introducís ', ' corregís ', ' distinguís ', ' admitís ', ' hervís ', ' asistís ', ' aludís ', ' desmentís ', ' destruís ', ' ingerís ', ' sumergís ', ' repartís ', ' traducís ', ' transferís ', ' teñís ', ' escupís ', ' intuís ', ' agredís ', ' acudís ', ' distribuís ', ' disminuís ', ' contradecís ', ' debatís ', ' intervenís ', ' pulís ', ' fingís ', ' residís ', ' evadís ', ' difundís ', ' añadís ', ' curtís ', ' interrumpís ', ' revivís ', ' sacudís ', ' reprimís ', ' requerís ', ' exhibís ', ' concluís ', ' batís ', ' exprimís ', ' imprimís ', ' infringís ', ' herís ', ' huís ', ' presidís ', ' omitís ', ' reproducís ', ' remitís ', ' sustituís ', ' restringís ', ' resumís ', ' seducís ', ' sobresalís ', ' prevenís ', ' impartís ', ' incurrís ', ' influís ', ' concurrís ', ' comprimís ', ' aducís ', ' derretís ', ' eludís ', ' freís ', ' expandís ', ' excluís ', ' escurrís ', ' esgrimís ', ' redescubrís ', ' prostituís ', ' provenís ', ' pudrís ', ' prescindís ', ' presentís ', ' oprimís ', ' irrumpís ', ' maldecís ', ' suplís ', ' revertís ', ' rescindís ', ' resentís ', ' transcribís ', ' transgredís ', ' ungís ', ' zambullís ', ' reincidís ', ' reñís ', ' suprimís ', ' sucumbís ', ' invadís ', ' predecís ', ' presumís ', ' proferís ', ' esparcís ', ' esculpís ', ' eximís ', ' impedís ', ' digerís ', ' desvivís ', ' devenís ', ' desdecís ', ' desistís ', ' deducís ', ' asentís ', ' afligís ', ' adscribís ', ' ceñís ', ' aturdís ', ' consentís ', ' constituís ', ' reís ', ' subís ', ' pensá ', ' andá ', ' escuchá ', ' sacá ', ' fumá ', ' tomá ', ' confiá ', ' cerrá ', ' mirá ', ' transmitá ', ' preguntá ', ' hablá ', ' jugá ', ' dependá ', ' aprovechá ', ' apretá ', ' contá ', ' calculá ', ' entrá ', ' disfrutá ', ' dejá ', ' firmá ', ' llamá ', ' probá ', ' pagá ', ' peleá ', ' tratá ', ' publicá ', ' recordá ', ' respetá ', ' tirá ', ' tocá ', ' votá ', ' saltá ', ' pintá ', ' mostrá ', ' mejorá ', ' pasá ', ' flotá ', ' interpretá ', ' guardá ', ' demostrá ', ' comprá ', ' educá ', ' descansá ', ' esperá ', ' estudiá ', ' exhalá ', ' contestá ', ' bailá ', ' bajá ', ' aggiorná ', ' mandá ', ' tenés ', ' podés ', ' querés ', ' sabés ', ' hacés ', ' estés ', ' creés ', ' ponés ', ' conocés ', ' debés ', ' entendés ', ' vés ', ' prevés ', ' rehacés ', ' retorcés ', ' escrachés ', ' naturalicés ', ' contraés ', ' sumés ', ' robés ', ' tachés ', ' sabés ', ' singularités ', ' transités ', ' retrotaés ', ' recompensés ', ' protejés ', ' candidateés ', ' entristecés ', ' observés ', ' mantienés ', ' mereés ', ' creés ', ' ponés ', ' debés ', ' necesitás ', ' hablás ', ' quedás ', ' pensás ', ' llegás ', ' dejás ', ' empezás ', ' mirás ', ' contás ', ' esperás ', ' preparás ', ' comparás ', ' anotás ']
    Tokens['Formality'] = {}
    Tokens['Formality']['words2check'] = [ "Boluda", "boluda", "Boludo", "boludo", "Che,", "Che!", "pelotudo ", " mierda ", " puta ", " carajo ", " loco ", " re ", " cagaste ", " jaja ", " chorro ", " imbécil "]
    Tokens['Formality']['RefWords'] = [" departamento ", " investigación ", " funcionario ", " resolución ", " denuncia ", " reforma ", " intervención ", " persecución ", " correspondiente ", " fiscal ", " desobediencia ", " federal ", " procedimiento ", " administrativa "]
    Tokens['Formality2'] = {}
    Tokens['Formality2']['words2check'] = ["amo' ", " ta ", " pa'", " facu ", " finde ", " cumple ", " vos ", "omo estás?", "omo estas?", "Che,", "Che!", " q "]
    Tokens['Formality2']['RefWords'] = ["amos ", " está ", " para ", " facultad ", "fin de semana", " cumpleaños", " usted ", "omo está?", "uenos días", " que "]
    Tokens['Formality3'] = {}
    Tokens['Formality3']['words2check'] = [ "aaa", "mmm", "ooo", "uuu", "eee"]
    Tokens['Formality3']['RefWords'] = ["a", "m", "o", "e", "u"]
    Tokens['Formality4'] = {}
    Tokens['Formality4']['words2check'] = ["porfa ", " cumple ", " vacas ", " facu ", " finde ", " cumple ", " tranqui ", " q ", " cole ", " bici ", " compi ", " peli ", " profe ", "progre ", "boli ", " compu ", "pelu ", "Uni", " U ", ]
    Tokens['Formality4']['RefWords'] = ["por favor ", " cumpleaños ", " vacaciones ", " facultad ", "fin de semana", " tranquillo", " tranquilla "," que ", "colegio", "bicicleta", "compañer", "película", "profesor", "progress", "bolígrafo", "computador", "peluquería", "Universidad"]
    Tokens['Formality5'] = {}
    Tokens['Formality5']['words2check'] = [" feca ", "quilombo", "piola", "cagatearse", "cagatina", "cagar", "boluda", " boludo", " Bolud", "puta", "mierda", " vos ", " te ", " tu ", " Tu ", " tú ", " Tú ", " cheta ", " cheto ", "cagada", "porfa ", " cumple ", " vacas ", " facu ", " finde ", " cumple ", " tranqui ", " q ", " cole ", " bici ", " compi ", " peli ", " profe ", "progre ", "boli ", " compu ", "pelu ", "Uni", " U "]
    Tokens['Formality5']['RefWords'] = ["café", "escándalo", "comprensiva", "sentir miedo", "Diarrea", "perjudicar", "estúpid", "por favor ", " señor", "Señor", " Usted ", "prostituta", "excremento", "usted ", " esnob ", " cumpleaños ", " vacaciones ", " facultad ", "fin de semana", " tranquillo", " tranquilla "," que ", "colegio", "bicicleta", "compañer", "película", "profesor", "progress", "bolígrafo", "computador", "peluquería", "Universidad"]
    Tokens['Formality6'] = {}
    Tokens['Formality6']['words2check'] = [" feca ", "quilombo", "piola", "cagatearse", "cagatina", "cagar", "boluda", " boludo", " Bolud", " culo ", "puta", "mierda"]
    Tokens['Formality6']['RefWords'] = ["café", "escándalo", "comprensiva", "sentir miedo", "Diarrea", "perjudicar", "estúpid", "por favor ", " señor", "Señor", " trasero ", "prostituta", "excremento"]
    Tokens['GenderNeutral'] = {}
    Tokens['GenderNeutral']['words2check'] =["rxs ", "r@s ", "todx", "tod@", "unx ", "un@ ", "nxs", "n@s", "lxs", "l@s", "in@ ", "inx ", "much@", "muchx", "amigues", "amig@", "amigx", "l@ ", "lx ", "s/-a ", "s/a ", "r/-a ", "r/a ", "chiques", "amigues ", "lxs", "l@s"]
    Tokens['GenderNeutral']['RefWords'] = ["ina ", "ino ", "inas ", "inos ", "los ", "las ", " el ", " la ", "chicos", "amigos ", "chicas", "amigas ", "otro", "otra", "mucha", "mucho", " un ", " una ", "unos ", "unas "]
    Tokens['Locations'] = {}
    Tokens['Locations']['Barrios_Olga'] = ["Agronomía", "Almagro", "Balvanera", "Barracas", "Belgrano", "Boedo", "Palermo", "Boca", "Caballito", "Chacarita", "Coghlan", "Colegiales", "Construción", "Flor", "Patern", "Liniers", "Monte Castro", "Montserrat", "Pompeya", "Parque", "Puerto", "Recoleta", "Retiro", "San Telmo", "San Nicol", "Villa", "Vélez"]
    Tokens['Locations']['Description'] = 'Names of Neighborhoods in Buenos Aires'
    Tokens['Locations']['Barrios_Wikipedia'] = ['Agronomía', 'Almagro', 'Balvanera', 'Barracas', 'Belgrano', 'Boedo', 'Caballito', 'Chacarita', 'Coghlan', 'Colegiales', 'Constitución', 'Flores', 'Floresta', 'La Boca', 'La Paternal', 'Liniers', 'Mataderos', 'Monserrat', 'Monte Castro', 'Nueva Pompeya', 'Núñez', 'Palermo', 'Parque Avellaneda', 'Parque Chacabuco', 'Parque Chas', 'Parque Patricios', 'Puerto Madero', 'Recoleta', 'Retiro', 'Saavedra', 'San Cristóbal', 'San Nicolás', 'San Telmo', 'Vélez Sársfield', 'Versalles', 'Villa Crespo', 'Villa del Parque', 'Villa Devoto', 'Villa General Mitre', 'Villa Lugano', 'Villa Luro', 'Villa Ortúzar', 'Villa Pueyrredón', 'Villa Real', 'Villa Riachuelo', 'Villa Santa Rita', 'Villa Soldati', 'Villa Urquiza']    
    Tokens['Locations']['Barrios_Wikipedia_Tokens'] = ['Agronomía', 'Agronomia', 'Almagro', 'Balvanera', 'Barracas', 'Belgrano', 'Boedo', 'Caballito', 'Chacarita', 'Coghlan', 'Colegiales', 'Constitución', 'Constitucion', 'Flores', 'Floresta', 'Boca', 'Paternal', 'Liniers', 'Mataderos', 'Monserrat', 'Monte', 'Castro', 'Nueva', 'Pompeya', 'Núñez', 'Núnez', 'Nuñez', 'Nunez', 'Palermo', 'Parque', 'Avellaneda', 'Chacabuco', 'Chas', 'Patricios', 'Puerto', 'Madero', 'Recoleta', 'Retiro', 'Saavedra', 'San', 'Cristóbal', 'Cristobal', 'Nicolás', 'Nicolas', 'Telmo', 'Vélez', 'Velez', 'Sársfield', 'Sarsfield', 'Versalles', 'Villa', 'Crespo', 'Devoto', 'Mitre', 'Lugano', 'Luro', 'Ortúzar', 'Ortuzar', 'Pueyrredón', 'Pueyrredon', 'Real', 'Riachuelo', 'Santa', 'Rita', 'Soldati', 'Urquiza']    
    Tokens['Locations']['FromCABA'] = []
    Tokens['Instagram'] = {}
    Tokens['Instagram']['words2check'] = ["Just posted "]
    Tokens['Instagram']['RefWords'] = ["Acaba de publicar "]
    Tokens['GenderNeutral'] = {}
    Tokens['GenderNeutral']['words2check'] = ["rxs ", "r@s ", "todx", "tod@", "unx ", "un@ ", "nxs", "n@s", "lxs", "l@s", "in@ ", "inx ", "much@", "muchx", "amigues", "amig@", "amigx", "l@ ", "lx ", "s/-a ", "s/a ", "r/-a ", "r/a ", "chiques", "amigues ", "lxs", "l@s"]
    Tokens['GenderNeutral']['RefWords'] = ["ina ", "ino ", "inas ", "inos ", "los ", "las ", " el ", " la ", "chicos", "amigos ", "chicas", "amigas ", "otro", "otra", "mucha", "mucho", " un ", " una ", "unos ", "unas "]
    Tokens['Null_vb'] = {}
    Tokens['Null_vb']['words2check'] = ["v"]
    Tokens['Null_vb']['RefWords'] = ["b"]
    Tokens['Null_loslas'] = {}
    Tokens['Null_loslas']['words2check'] = [" los "]
    Tokens['Null_loslas']['RefWords'] = [" las "]
    Tokens['BocaPalermo'] = {}
    Tokens['BocaPalermo']['words2check'] = ["a Boca "]
    Tokens['BocaPalermo']['RefWords'] = [" Palermo "]
    Tokens['Stadiums'] = {}
    Tokens['Stadiums']['words2check'] = ['en Estadio "Monumental" Antonio']
    Tokens['Stadiums']['RefWords'] = ["Bombonera"]
    Tokens['TangoFutbol'] = {}
    Tokens['TangoFutbol']['words2check'] = ["tango ", "Tango "]
    Tokens['TangoFutbol']['RefWords'] = ["fútbol ", "Fútbol "]
    return Tokens[name]


def NDig(coord):
    return len(str(coord).split('.')[1])

def GetMinDigits(GeoData):
	MinLonDigits = min([NDig(x) for x in GeoData[0]])
	MinLatDigits = min([NDig(x) for x in GeoData[1]])
	return min([MinLonDigits, MinLatDigits])

def GetRefWordUserInds(ID_UserType_Inds, ID_User_Data, RefWords, words2check):
    print('Finding User indices filtered by tokens...')
    print('Total Number of users: {}. Number of User-Type Users: {}'.format(len(ID_User_Data[0]),len(ID_UserType_Inds)))
    #Get Ref and Word User Indices
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_UserType_ref_Inds = [] # List of indices associated with ref users satisfying full search condition
    ID_UserType_word_Inds = [] # List of indices associated with word users satisfying full search condition
    iInd = 0
    ID_texts = ID_User_Data[4] 
    for Ind in ID_UserType_Inds:
        for text in ID_texts[Ind]:
            if any(wrd in text for wrd in RefWords):
                ID_UserType_ref_Inds.append(Ind)
            if any(wrd in text for wrd in words2check):
                ID_UserType_word_Inds.append(Ind)
        iInd += 1
        if True and iInd%10000 == 0:
            print('Users Processed: ', iInd)
    ID_UserType_ref_Inds = list(set(ID_UserType_ref_Inds))
    ID_UserType_word_Inds = list(set(ID_UserType_word_Inds))
    ID_UserType_2D_Inds = list(set(ID_UserType_ref_Inds) & set(ID_UserType_word_Inds))
    return ID_UserType_ref_Inds, ID_UserType_word_Inds, ID_UserType_2D_Inds
    #ID_UserType_ref_Inds, ID_UserType_word_Inds, ID_UserType_2D_Inds = GetRefWordUserInds(ID_UserType_Inds, ID_User_Data, RefWords, words2check)

def GetLocationUserTypeInds(ID_UserType_Inds, ID_User_Data, LocationTokens):
    print('Finding User indices filtered by locations...')
    print('Total Number of users: {}. Number of User-Type Users: {}'.format(len(ID_User_Data[0]),len(ID_UserType_Inds)))
    #Get Ref and Word User Indices
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    ID_UserType_Loc_Inds = [] # List of indices associated with location users
    iInd = 0
    ID_locations = ID_User_Data[8] 
    for Ind in ID_UserType_Inds:
        for location in ID_locations[Ind]:
            if location in LocationTokens:
                ID_UserType_Loc_Inds.append(Ind)
        iInd += 1
        if True and iInd%10000 == 0:
            print('Users Processed: ', iInd)
    ID_UserType_Loc_Inds = list(set(ID_UserType_Loc_Inds))
    return ID_UserType_Loc_Inds
    #ID_UserType_Loc_Inds = GetLocationUserTypeInds(ID_UserType_Inds, ID_User_Data, LocationTokens)

def GetElts(List_User_Elts):
    print('Collecting Elements from all users...')
    print('Number of Users: {}.'.format(len(List_User_Elts)))
    #List_User_Elts structure: List_User_Elts[i] = [IDs[i], ID_counts[i], ID_lons[i], ID_lats[i], ID_texts[i], ID_dates[i], ID_times[i], ID_names[i], ID_locations[i]]
    NumElts = len(List_User_Elts[0])
    ID_lons = [x[2] for x in List_User_Elts]
    ID_lats = [x[3] for x in List_User_Elts]
    ID_texts = [x[4] for x in List_User_Elts]
    ID_dates = [x[5] for x in List_User_Elts]
    ID_times = [x[6] for x in List_User_Elts]
    ID_names = [x[7] for x in List_User_Elts]
    ID_locations = [x[8] for x in List_User_Elts]
    return ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations
    #ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations = GetElts(List_User_Elts)

def GetUserData(List_User_Elts):
    print('Collecting data from all users...')
    print('Number of Users: {}.'.format(len(List_User_Elts)))
    #List_User_Elts: List_User_Elts[i] = [IDs[i], ID_counts[i], ID_lons[i], ID_lats[i], ID_texts[i], ID_dates[i], ID_times[i], ID_names[i], ID_locations[i]]
    NumElts = len(List_User_Elts[0])
    ID_User_Data = [[] for i in range(NumElts-2)]
    iElt = 0
    for Elt in List_User_Elts:
        for i in range(2,NumElts):
            ID_User_Data[i] = ID_User_Data[i] + Elt[i]
        iElt += 1
        if True and iInd%10000 == 0:
            print('Users Processed: ', iElt)
    return ID_User_Data
    #ID_User_Data = GetUserData(List_User_Elts)

def GetUserIDInds(ID_User_Data, Selected_IDs):
	print('Finding User indices filtered by User ID...')
	print('Total Number of users: {}. Number of User-Type Users: {}'.format(len(ID_User_Data[0]),len(Selected_IDs)))
	#ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
	ID_Inds = [] # List of indices associated with selected User IDs
	IDs = ID_User_Data[0]
	for i in range(len(IDs)):
		if IDs[i] in Selected_IDs:
			ID_Inds.append(i)
		if i%10000 == 0:
			print('Users Processed: {}'.format(i))
	return ID_Inds
	#ID_Inds = GetUserIDInds(ID_User_Data, Selected_IDs)

def FilterByUserInds(ID_User_Data, ID_UserType_Inds):
    print('Filtering data from {} selected users out of a total of {}...'.format(len(ID_User_Data[0]),len(ID_UserType_Inds)))
    #ID_User_Data structure: ID_User_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    NumElts = len(ID_User_Data)
    ID_UserType_Data = [[] for i in range(NumElts)]
    iInd = 0
    for Ind in ID_UserType_Inds :
        for Elt in range(NumElts):
            ID_UserType_Data[Elt].append(ID_User_Data[Elt][Ind])
        iInd += 1
        if True and iInd%10000 == 0:
            print('Users Processed: ', iInd)
    return ID_UserType_Data
    #ID_UserType_Data = FilterByUserInds(ID_User_Data, ID_UserType_Inds)

def CollectData(ID_UserType_Data):
    starttime = datetime.now()
    print('Collecting data from {} users...'.format(len(ID_UserType_Data[0])))
    #ID_UserType_Data structure: ID_UserType_Data = [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations]
    NumElts = len(ID_UserType_Data)
    GeoData = [[] for i in range(NumElts-2)]
    for i in range(len(ID_UserType_Data[0])):
        for Elt in range(NumElts-2):
            for x in ID_UserType_Data[Elt+2][i]:
                GeoData[Elt].append(x)
        if True and i%10000 == 0:
            print('Users Processed: {}'.format(i))
    print('Tweets processed: {}'.format(len(GeoData[0])))
    endtime = datetime.now()
    dt = endtime - starttime
    print('Output: GeoData = [lons, lats, texts, dates, times, names, locations]. Execution time: {}'.format(dt))
    return GeoData
    #GeoData = CollectData(ID_UserType_Data)

def FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter):
    if Filter:
        starttime = datetime.now()
        print('Filtering tweets by digits of precision and time frame...')
        print('MinDigits = {}. TimeFrame = {}'.format(MinDigits,TimeFrame))
        print('Number of tweets: {}.'.format(len(GeoData[0])))
        GeoData_f = [[] for x in GeoData]
        ix = 0
        C_NDigit = True # By default, count Tweets with all numbers of digits of precision
        C_TimeFrame = True # By default, count Tweets from all time periods
        for i in range(len(GeoData[0])):
            datum = [lon,lat,text,date,time,name,location] = [x[i] for x in GeoData]
            if MinDigits > 0:
                C_NDigit = NDig(lon) >= MinDigits and NDig(lat) >= MinDigits
            elif MinDigits < 0:
                C_NDigit = NDig(lon) == -MinDigits and NDig(lat) == -MinDigits
            if TimeFrame == 'Pre06.2019':
                Y,M,D = getYMDnum(date)
                C_TimeFrame = Y < 2019 or (Y == 2019 and M <= 6)
            elif TimeFrame == 'Post06.2019':
                Y,M,D = getYMDnum(date)
                C_TimeFrame = Y > 2019 or (Y == 2019 and M > 6)
            elif TimeFrame == 'AllTimes':
                C_TimeFrame = True
            else:
                C_TimeFrame = False
                print('Time frame not recognized!')
            if C_TimeFrame and C_NDigit:
                for j in range(len(GeoData_f)):
                    GeoData_f[j].append(datum[j])
            ix += 1
            if True and ix%100000 == 0:
                print('Tweets Processed: {}. Digit Count Satisfied: {}. TimeFrame satisfied: {}'.format(ix,C_NDigit,C_TimeFrame))
        print('Number of Tweets after filtering: {}'.format(len(GeoData_f[0])))
        endtime = datetime.now()
        dt = endtime - starttime
        print('Output: GeoData_f = [lons, lats, texts, dates, times, names, locations]. Execution time: {}'.format(dt))
        return GeoData_f
    else:
        print('Filtering bypassed.')
        return GeoData
    #GeoData_f = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter)

def FilterByTokens(GeoData, tokens, Filter, Include):
    if Filter:
        starttime = datetime.now()
        print('Filtering tweets by tokens...')
        print('Number of tokens: {}. Number of tweets: {}.'.format(len(tokens),len(GeoData[0])))
        GeoData_f = [[] for x in GeoData]
        for i in range(len(GeoData[0])):
            datum = [lon,lat,text,date,time,name,location] = [x[i] for x in GeoData]
            if Include:
                if any([x in text for x in tokens]):
                    for j in range(len(GeoData_f)):
                        GeoData_f[j].append(datum[j])
            else:
                if not any([x in text for x in tokens]):
                    for j in range(len(GeoData_f)):
                        GeoData_f[j].append(datum[j])
            if True and i%10000 == 0:
                print('Tweets Processed: {}'.format(i))
        print('Number of Tweets after filtering: {}'.format(len(GeoData_f[0])))
        endtime = datetime.now()
        dt = endtime - starttime
        print('Output: GeoData_f = [lons, lats, texts, dates, times, names, locations]. Execution time: {}'.format(dt))
        return GeoData_f
    else:
        print('Filtering bypassed.')
        return GeoData
    #GeoData_f = FilterByTokens(GeoData, tokens, Filter, Include)

def FilterByLocations(GeoData, LocationTokens, Filter, Include):
    if Filter:
        starttime = datetime.now()
        print('Filtering tweets by locations...')
        print('Number of locations: {}. Number of tweets: {}.'.format(len(LocationTokens),len(GeoData[0])))
        GeoData_f = [[] for x in GeoData]
        for i in range(len(GeoData[0])):
            datum = [lon,lat,text,date,time,name,location] = [x[i] for x in GeoData]
            if Include:
                if location in LocationTokens:
                    for j in range(len(GeoData_f)):
                        GeoData_f[j].append(datum[j])
            else:
                if not location in LocationTokens:
                    for j in range(len(GeoData_f)):
                        GeoData_f[j].append(datum[j])
            if True and i%10000 == 0:
                print('Tweets Processed: {}'.format(i))
        print('Number of Tweets after filtering: {}'.format(len(GeoData_f[0])))
        endtime = datetime.now()
        dt = endtime - starttime
        print('Output: GeoData_f = [lons, lats, texts, dates, times, names, locations]. Execution time: {}'.format(dt))
        return GeoData_f
    else:
        print('Filtering bypassed.')
        return GeoData
    #GeoData_f = FilterByTokens(GeoData, LocationTokens, Filter, Include)

def FilterByExtent(GeoData, extent, Filter):
    if Filter:
        starttime = datetime.now()
        print('Filtering tweets by extent...')
        print('extent: {}. Number of tweets: {}.'.format(extent,len(GeoData[0])))
        GeoData_f = [[] for x in GeoData]
        Longmin,Longmax = min(extent[0],extent[1]), max(extent[0],extent[1])
        Latmin,Latmax = min(extent[2],extent[3]), max(extent[2],extent[3])
        for i in range(len(GeoData[0])):
            datum = [lon,lat,text,date,time,name,location] = [x[i] for x in GeoData]
            if Longmin <= lon <= Longmax and Latmin <= lat <= Latmax:
                for j in range(len(GeoData_f)):
                    GeoData_f[j].append(datum[j])
            if True and i%100000 == 0:
                print('Tweets Processed: {}'.format(i))
        print('Number of Tweets after filtering: {}'.format(len(GeoData_f[0])))
        endtime = datetime.now()
        dt = endtime - starttime
        print('Output: GeoData_f = [lons, lats, texts, dates, times, names, locations]. Execution time: {}'.format(dt))
        return GeoData_f
    else:
        print('Filtering bypassed.')
        return GeoData
    #GeoData_f = FilterByExtent(GeoData, extent, Filter)

def SaveText(text, filepath):
	with open(filepath, 'w', encoding='utf-8') as f:
		print(text, file = f)
def now():
	DateStr = datetime.now().strftime('%Y_%m%d_')
	return DateStr

def datetimestr():
    return datetime.now().strftime('%Y_%m%d_%H%M%S')

def GetExtent(name):
    EXTENT = {}
    EXTENT['US'] = [-124.7844079,-66.9513812, 24.7433195, 49.3457868] # Contiguous US bounds
    EXTENT['NYC'] = [-74.257159,-73.699215,40.495992,40.915568] # NYC bounds
    EXTENT['Italy'] = [6.7499552751, 36.619987291, 18.4802470232, 47.1153931748]# Italy
    EXTENT['Spain'] = [35.946850084, 43.7483377142,  -9.39288367353, 3.03948408368]#Spain
    EXTENT['DR&PR'] = [-71.9451120673, -68.3179432848, 19.8849105901, 18.0]# Dominican Rep+Puerto Rico.
    EXTENT['France'] = [51.1485061713, -54.5247541978, 2.05338918702, 9.56001631027]#France
    EXTENT['Mexico'] = [-117.12776, -86.811982388, 14.5388286402, 32.72083] #Mexico
    EXTENT['Germany'] = [47.3024876979, 54.983104153, 5.98865807458, 15.0169958839]# Germany
    EXTENT['South America'] = [-90, -30, -50, 10] # South America
    EXTENT['Buenos Aires'] = [-58.969, -57.844, -34.339, -35.037] # Full Buenos Aires
    EXTENT['Main Buenos Aires'] = [-58.531725, -58.355148, -34.538162, -34.705446] # Main Buenos Aires
    EXTENT['Exact CABA'] = [-58.531506, -58.335144, -34.526514, -34.705372] # Administrative borders of Buenos Aires
    EXTENT['Center Buenos Aires'] = [-58.421637, -58.331714, -34.583043, -34.661079] # Center Buenos Aires
    EXTENT['Province of Buenos Aires'] = [-59.90, -56.844, -34.339, -35.037]# province of Buenos Aires
    EXTENT['BARef'] = [-58.365923, -58.464860, -34.622559, -34.534517]
    EXTENT['Corners of BA'] = [-58.530946, -58.335169, -34.526665, -34.705089] # Reference for comparing Cartopy and Google Maps coord systems
    EXTENT['World Map'] = [-150, 150, -60, 70]# most of world
    EXTENT['US and South America'] = [-130, -30, -45, 40]
    EXTENT['Cordoba'] = [-64.272, -64.105, -31.470, -31.347] # Cordoba
    EXTENT['Greater Paris'] = [1.989885, 2.619273, 49.068653, 48.58074] # Greater Paris
    EXTENT['Paris Center'] = [2.256853, 2.416401, 48.903854, 48.816817] # Center circle of Paris
    EXTENT['Sao Paulo'] =[-46.825290,-46.325290, -23.3,-23.9533773]
    return EXTENT[name]

def MatrixBinElts(lons, lats, extent, NumLongBins, NumLatBins, Elts):
    print('Binning Tweets into {}x{} bins in extent {}'.format(NumLongBins,NumLatBins,extent))
    goElts = Elts != [] #Check if Elts has anything in it
    Longmin,Longmax = min(extent[0],extent[1]), max(extent[0],extent[1])
    Latmin,Latmax = min(extent[2],extent[3]), max(extent[2],extent[3])
    LongBinSize, LatBinSize = abs(Longmax - Longmin)/NumLongBins, abs(Latmax - Latmin)/NumLatBins
    LongBinLimits = [Longmin + i*LongBinSize for i in range(NumLongBins+1)]
    LatBinLimits = [Latmin + i*LatBinSize for i in range(NumLatBins+1)]
    #LongBinLimits = np.arange(Longmin, Longmax + LongBinSize, LongBinSize).tolist()
    #LatBinLimits = np.arange(Latmin, Latmax + LatBinSize, LatBinSize).tolist()
    LongBinCenters = [(LongBinLimits[i]+LongBinLimits[i+1])/2 for i in range(NumLongBins)]
    LatBinCenters = [(LatBinLimits[i]+LatBinLimits[i+1])/2 for i in range(NumLatBins)]
    Matrix = [[0 for x in range(NumLongBins)] for y in range(NumLatBins)] # Initializes Matrix
    if goElts:
        Matrix_Elts = [[[] for x in range(NumLongBins)] for y in range(NumLatBins)] # Initializes Matrix
    Matrix_Lons = [LongBinCenters for y in range(NumLatBins)] # Create a matrix with the longitude coordinate of each bin center
    Matrix_Lats = [[LatBinCenters[y] for x in range(NumLongBins)] for y in range(NumLatBins)]# Create a matrix with the latitude coordinate of each bin center
    # Perform binning
    printfreq = 100000
    for i in range(len(lons)):
        if Longmin <= lons[i] <= Longmax and Latmin <= lats[i] <= Latmax:
            x = np.searchsorted(LongBinLimits, lons[i], side="left") - 1
            y = np.searchsorted(LatBinLimits, lats[i], side="left") - 1
            #idx = findbin(LongBinLimits, LatBinLimits, lons[i], lats[i]) #find the correct matrix element to put the tweet
            idx = [x,y]
            Matrix[idx[1]][idx[0]]+=1
            if goElts:
                Matrix_Elts[idx[1]][idx[0]].append(Elts[i])
            if i%printfreq == 0:
                print('Tweets processed: {}'.format(i))
    Matrix_flat = list(itertools.chain(*Matrix))
    Matrix_Lons_flat = list(itertools.chain(*Matrix_Lons))
    Matrix_Lats_flat = list(itertools.chain(*Matrix_Lats))
    if goElts:
        Matrix_Elts_flat = list(itertools.chain(*Matrix_Elts))
    else:
        Matrix_Elts_flat = []
    return Matrix_Lons_flat, Matrix_Lats_flat, Matrix_flat, Matrix_Elts_flat

def findbin(LongBins, LatBins, Long, Lat):
    if min(LongBins) <= Long <= max(LongBins):
        x = np.searchsorted(LongBins, Long, side="left") - 1
    else:
        x = -1
    if min(LatBins) <= Lat <= max(LatBins):
        y = np.searchsorted(LatBins, Lat, side="left") - 1
    else:
        y = -1
    return [x,y]

def GetBinPhysSize(extent, NumLonBins, NumLatBins):
	LonRange = abs(extent[0] - extent[1])
	LatRange = abs(extent[2] - extent[3])
	LatCenter = (extent[2] + extent[3])/2
	dlon = LonRange/NumLonBins
	dlat = LatRange/NumLatBins
	R_Earth = 6371000 #Earth's radius in meters
	dx = R_Earth * dlon * (np.pi/180) * np.cos(np.pi * LatCenter/180)
	dy = R_Earth * dlat * (np.pi/180)
	return dx, dy

def NumTweets(lons, Inds):
	return sum([len(lons[x]) for x in Inds])

def GetWmR(counts_word, counts_ref):
    Sum_word = sum(counts_word)
    Sum_ref = sum(counts_ref)
    WmR = [counts_word[i]/Sum_word - counts_ref[i]/Sum_ref for i in range(len(counts_word))]
    return WmR

def GetWmRabs(counts_word, counts_ref):
	WmRabs = [counts_word[i] - counts_ref[i] for i in range(len(counts_word))]
	return WmRabs

def GetPosNegData(Lons, Lats, WordsMinusRef, ElimZeros):
    WmR_pos, WmR_neg, WmR_pos_Lons,  WmR_neg_Lons, WmR_pos_Lats, WmR_neg_Lats = [], [], [], [], [], []
    for i in range(len(WordsMinusRef)):
        WmR = WordsMinusRef[i]
        if WmR > 0 or ((not ElimZeros) and WmR == 0 ): #include zeros if ElimZeros is False
            WmR_pos.append(WmR)
            WmR_pos_Lons.append(Lons[i])
            WmR_pos_Lats.append(Lats[i])
        if WmR < 0:
            WmR_neg.append(WmR)
            WmR_neg_Lons.append(Lons[i])
            WmR_neg_Lats.append(Lats[i])
    WmR_pn_data = [WmR_pos_Lons, WmR_pos_Lats, WmR_pos, WmR_neg_Lons, WmR_neg_Lats, WmR_neg]
    return WmR_pn_data

def GetPosNegData_Zeros(Lons, Lats, WordsMinusRef):
    WmR_pos, WmR_neg, WmR_pos_Lons,  WmR_neg_Lons, WmR_pos_Lats, WmR_neg_Lats, WmR_0_Lons, WmR_0_Lats = [], [], [], [], [], [], [], []
    for i in range(len(WordsMinusRef)):
        WmR = WordsMinusRef[i]
        if WmR > 0:
            WmR_pos.append(WmR)
            WmR_pos_Lons.append(Lons[i])
            WmR_pos_Lats.append(Lats[i])
        if WmR < 0:
            WmR_neg.append(WmR)
            WmR_neg_Lons.append(Lons[i])
            WmR_neg_Lats.append(Lats[i])        
        if WmR == 0:
            WmR_0_Lons.append(Lons[i])
            WmR_0_Lats.append(Lats[i])
    WmR_pn_data = [WmR_pos_Lons, WmR_pos_Lats, WmR_pos, WmR_neg_Lons, WmR_neg_Lats, WmR_neg, WmR_0_Lons, WmR_0_Lats]
    return WmR_pn_data

def FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_Word, GeoData_Ref, ElimZeros):
    lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_Ref[0], GeoData_Ref[1], extent, NumBins, NumBins, Elts=[])
    lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_Word[0], GeoData_Word[1], extent, NumBins, NumBins, Elts=[])
    if Absolute:
        WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
        return GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
    else:
        WmR = GetWmR(counts_word_M, counts_ref_M)
        return GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
    #ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_Word, GeoData_Ref, ElimZeros)

def AspectRatio(extent):
	lat0 = (extent[2]+extent[3])/2
	AR = (extent[2]-extent[3])/((extent[1]-extent[0])*np.cos(-lat0*np.pi/180))
	return AR

def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

def pltposneg(Show, posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree())
    if Show:
        plt.show()
    return fig

def pltposneg_format(GraphFormat, Show, posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    if GraphFormat['FigSize'] == []:
        fig = plt.figure() # open matplotlib figure
    else:
        fig = plt.figure(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1])) # open matplotlib figure
    TitleFontSize, AxesFontSize = GraphFormat['TitleFontSize'], GraphFormat['AxesFontSize']
    xtics,ytics = GraphFormat['NumTics']
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=TitleFontSize)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],xtics),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],ytics)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
    ax1.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    #FigSize = plt.rcParams['figure.figsize']
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree())
    if GraphFormat['tight']:
        plt.tight_layout()
    if Show:
        plt.show()
    return fig

def pltposneg_Zeros(Show, posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, nullcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    nulllons = np.array(posnegdata[6])
    nulllats = np.array(posnegdata[7])
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
        nullcirclesizes = [-circ_scale for x in nulllons]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
        nullcirclesizes = [0 for x in nulllons]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.scatter(x=nulllons, y=nulllats, color=nullcolor, s=nullcirclesizes, alpha=alpha,transform=crs.PlateCarree())
    if Show:
        plt.show()
    return fig

def pltposneg_square_old(Show, MarkerScale, NumBins,posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    FigSize = plt.rcParams['figure.figsize']
    A = AspectRatio(extent)
    #MarkerScale = 77 # for Buenos Aires
    #MarkerScale = 58 # for Main Buenos Aires
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
    elif circ_scale == 0:
        poscirclesizes = [(MarkerScale*FigSize[0]/NumBins)**2 for x in posfreq]
        negcirclesizes = [(MarkerScale*FigSize[0]/NumBins)**2 for x in negfreq]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
    verts = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] #relative coordinates of marker
    RectVerts = [[x[0],x[1]*A] for x in verts]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    if Show:
        plt.show()
    return fig

def pltposneg_square_format(GraphFormat, Show, MarkerScale, NumBins,posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    if GraphFormat['FigSize'] == []:
        fig = plt.figure() # open matplotlib figure
    else:
        fig = plt.figure(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1])) # open matplotlib figure
    TitleFontSize, AxesFontSize = GraphFormat['TitleFontSize'], GraphFormat['AxesFontSize']
    xtics,ytics = GraphFormat['NumTics']
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=TitleFontSize)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],xtics),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],ytics)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
    ax1.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    FigSize = plt.rcParams['figure.figsize']
    A = AspectRatio(extent)
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
    elif circ_scale == 0:
        poscirclesizes = [(MarkerScale*FigSize[0]/NumBins)**2 for x in posfreq]
        negcirclesizes = [(MarkerScale*FigSize[0]/NumBins)**2 for x in negfreq]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
    verts = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] #relative coordinates of marker
    RectVerts = [[x[0],x[1]*A] for x in verts]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    if GraphFormat['tight']:
        plt.tight_layout()
    if Show:
        plt.show()
    return fig

def pltposneg_square_format_fill(GraphFormat, Show, NumBins,posnegdata, extent, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    if GraphFormat['FigSize'] == []:
        fig = plt.figure() # open matplotlib figure
    else:
        fig = plt.figure(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1])) # open matplotlib figure
    TitleFontSize, AxesFontSize = GraphFormat['TitleFontSize'], GraphFormat['AxesFontSize']
    xtics,ytics = GraphFormat['NumTics']
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=TitleFontSize)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],xtics),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],ytics)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
    ax1.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    BinSizeLons, BinSizeLats = abs(extent[0]-extent[1])/NumBins, abs(extent[2]-extent[3])/NumBins
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    for i in range(len(poslons)):
	    x0,x1 = poslons[i] - BinSizeLons/2, poslons[i] + BinSizeLons/2
	    y0,y1 = poslats[i] - BinSizeLats/2, poslats[i] + BinSizeLats/2
	    plt.fill_between([x0,x1],[y0,y0],[y1,y1], color=poscolor, linewidth=0, alpha = alpha, transform=crs.PlateCarree())
    for i in range(len(neglons)):
	    x0,x1 = neglons[i] - BinSizeLons/2, neglons[i] + BinSizeLons/2
	    y0,y1 = neglats[i] - BinSizeLats/2, neglats[i] + BinSizeLats/2
	    plt.fill_between([x0,x1],[y0,y0],[y1,y1], color=negcolor, linewidth=0, alpha = alpha, transform=crs.PlateCarree())
    if GraphFormat['tight']:
        plt.tight_layout()
    if Show:
        plt.show()
    return fig

def pltposneg_square_new(Show, MarkerScale, posnegdata, extent, circ_scale, title, img, bdr, grd, poscolor, negcolor, alpha, osm_img):
    poslons = np.array(posnegdata[0])
    poslats = np.array(posnegdata[1])
    posfreq = np.array(posnegdata[2])
    neglons = np.array(posnegdata[3])
    neglats = np.array(posnegdata[4])
    negfreq = np.array(posnegdata[5])
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    FigSize = plt.rcParams['figure.figsize']
    NumLonBins, NumLatBins = GetNBins(extent, posnegdata)
    A = AspectRatio(extent)*NumLonBins/NumLatBins
    #MarkerScale = 77 # for Buenos Aires
    #MarkerScale = 58 # for Main Buenos Aires
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        poscirclesizes = [-circ_scale for x in posfreq]
        negcirclesizes = [-circ_scale for x in negfreq]
    elif circ_scale == 0:
        poscirclesizes = [(MarkerScale*FigSize[0]/NumLonBins)**2 for x in posfreq]
        negcirclesizes = [(MarkerScale*FigSize[0]/NumLonBins)**2 for x in negfreq]
    else:
        poscirclesizes = [abs(x)*circ_scale for x in posfreq]
        negcirclesizes = [abs(x)*circ_scale for x in negfreq]
    verts = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] #relative coordinates of marker
    RectVerts = [[x[0],x[1]*A] for x in verts]
    plt.scatter(x=poslons, y=poslats, color=poscolor, s=poscirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    plt.scatter(x=neglons, y=neglats, color=negcolor, s=negcirclesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    if Show:
        plt.show()
    return fig

def GetRedBlueCoord(WordsMinusRef, lons, lats):
    WmR_indices = [i for i in range(len(WordsMinusRef))]
    SortedWmR = sorted(zip(WordsMinusRef, lons, lats, WmR_indices))
    RSortedWmR = sorted(zip(WordsMinusRef, lons, lats, WmR_indices), reverse = True)
    BlueCoord = [[b,c] for a,b,c,d in SortedWmR]
    RedCoord = [[b,c] for a,b,c,d in RSortedWmR]
    BlueVals = [a for a,b,c,d in SortedWmR]
    RedVals = [a for a,b,c,d in RSortedWmR]
    BlueIndices = [d for a,b,c,d in SortedWmR]
    RedIndices = [d for a,b,c,d in RSortedWmR]
    return BlueCoord, RedCoord, BlueVals, RedVals, BlueIndices, RedIndices
    # Sample Command: BlueCoord_nonStatic, RedCoord_nonStatic, BlueVals_nonStatic, RedVals_nonStatic, BlueIndices_nonStatic, RedIndices_nonStatic = GetRedBlueCoord(ID_nonStatic_WmR, lons_nonStatic_ref, lats_nonStatic_ref)

def pltposnegmax(WordsMinusRef, lons, lats, N, extent, plottitle, ElimZeros, circ_scale):
    BlueCoord, RedCoord, BlueVals, RedVals, BlueIndices, RedIndices = GetRedBlueCoord(WordsMinusRef, lons, lats)
    MaxNVals = RedVals[0:N]+BlueVals[0:N]
    MaxNLons = [RedCoord[i][0] for i in range(N)] + [BlueCoord[i][0] for i in range(N)]
    MaxNLats = [RedCoord[i][1] for i in range(N)] + [BlueCoord[i][1] for i in range(N)]
    MaxN_pndata = GetPosNegData(MaxNLons, MaxNLats, MaxNVals, ElimZeros)
    pltposneg(MaxN_pndata, extent, circ_scale, title=plottitle, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.4, osm_img=osm_img)

def GetBinSize(extent, NumLongBins, NumLatBins):
	LongBinSize = abs(extent[0] - extent[1])/NumLongBins
	LatBinSize = abs(extent[2]-extent[3])/NumLatBins
	return LongBinSize, LatBinSize

def GetHist(anylist):
    Items = []
    Counts = []
    for x in anylist:
            if x not in Items:
                    Items.append(x)
                    Counts.append(1)
            else:
                    i = Items.index(x)
                    Counts[i] += 1
    Sorted_Hist = sorted(zip(Counts,Items), reverse = True)
    Sorted_Items = [a for b,a in Sorted_Hist]
    Sorted_Counts = [b for b,a in Sorted_Hist]
    return Sorted_Items, Sorted_Counts

def Sort2List(Counts, Items):
    Sorted_Hist = sorted(zip(Counts,Items), reverse = True)
    Sorted_Items = [a for b,a in Sorted_Hist]
    Sorted_Counts = [b for b,a in Sorted_Hist]
    return Sorted_Counts, Sorted_Items

def GetInds(tweet_IDs):
    Sorted_IDs = sorted(tweet_IDs)
    Inds = []
    ID_0 = Sorted_IDs[0]
    i=0
    for ID in Sorted_IDs:
        if ID != ID_0:
            Inds.append(i)
        ID_0 = ID
        i+=1
    return Inds

def QuickSort(RList):
        SRList = sorted(RList)
        ChInds = [i for i in range(1,len(SRList)) if SRList[i] != SRList[i-1]]
        LimInds = [0]+ChInds+[len(SRList)]
        BRList = [SRList[LimInds[i]:LimInds[i+1]-1] for i in range(len(LimInds)-1)]
        return BRList

def PlotPoint(TweetCoords, PointCoord, LongBinSize, LatBinSize, NumBinsPlot, NumBinsTweets, Title, Hist, pltcolor, Alpha):
	dlon_plot = abs(LongBinSize*NumBinsPlot)
	dlat_plot = abs(LatBinSize*NumBinsPlot)
	extent_plot = [PointCoord[0]-dlon_plot,PointCoord[0]+dlon_plot,PointCoord[1]-dlat_plot,PointCoord[1]+dlat_plot]
	dlon_Tweets = abs(LongBinSize*NumBinsTweets)
	dlat_Tweets = abs(LatBinSize*NumBinsTweets)
	extent_Tweets = [PointCoord[0]-dlon_Tweets,PointCoord[0]+dlon_Tweets,PointCoord[1]-dlat_Tweets,PointCoord[1]+dlat_Tweets]
	PlotCoords = []
	for i in range(len(TweetCoords[0])):
		lon, lat = TweetCoords[0][i], TweetCoords[1][i]
		if lon >= extent_Tweets[0] and lon <= extent_Tweets[1] and lat >= extent_Tweets[2] and lat <= extent_Tweets[3]:
			PlotCoords.append([lon, lat])
	if Hist:
		HistPlotCoords, freq = GetHist(PlotCoords)
		PlotLons = [x[0] for x in HistPlotCoords]
		PlotLats = [x[1] for x in HistPlotCoords]
		circ_scale = 1
		#Alpha = 0.3
	else:
		PlotLons = [x[0] for x in PlotCoords]
		PlotLats = [x[1] for x in PlotCoords]
		freq = [1 for x in PlotLons]
		circ_scale = -1
		#Alpha = 1
	mkscatplt2([PlotLons,PlotLats], np.array(freq), extent_plot, circ_scale, title=Title, img='y', bdr='y', grd='n', clr=pltcolor, cmap='bwr', alpha=Alpha, osm_img=osm_img)
	return PlotLons, PlotLats, freq, extent_plot, extent_Tweets

def SaveNTextsAndCollocations(N, directory, texts_word, texts_ref, texts_all, RedIndices, BlueIndices):
    for i in range(N):
        DateStr = datetime.now().strftime('%Y_%m%d_')
        SaveText(collocation(texts_word[RedIndices[i]], 50), directory + DateStr+'Red'+str(i+1)+'_Word Tweet Collocations.txt')
        SaveText(collocation(texts_ref[BlueIndices[i]], 50), directory + DateStr+'Blue'+str(i+1)+'_Ref Tweet Collocations.txt')
        SaveText(collocation(texts_all[RedIndices[i]], 500), directory + DateStr+'Red'+str(i+1)+'_All Tweet Collocations.txt')
        SaveText(collocation(texts_all[BlueIndices[i]], 500), directory + DateStr+'Blue'+str(i+1)+'_All Tweet Collocations.txt')
        SaveText(texts_word[RedIndices[i]], directory + DateStr+'Red'+str(i+1)+'_Word Tweet Texts.txt')
        SaveText(texts_ref[RedIndices[i]], directory + DateStr+'Red'+str(i+1)+'_Ref Tweet Texts.txt')
        SaveText(texts_all[RedIndices[i]], directory + DateStr+'Red'+str(i+1)+'_All Tweet Texts.txt')
        SaveText(texts_word[BlueIndices[i]], directory + DateStr+'Blue'+str(i+1)+'_Word Tweet Texts.txt')
        SaveText(texts_ref[BlueIndices[i]], directory + DateStr+'Blue'+str(i+1)+'_Ref Tweet Texts.txt')
        SaveText(texts_all[BlueIndices[i]], directory + DateStr+'Blue'+str(i+1)+'_All Tweet Texts.txt')

def GetTitle(TitleElts):
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    T_UserType = UserType+'Users'
    T_TimeFrame = TimeFrame
    if MinDigits > 0:
        T_Digits = str(MinDigits) + '+Digits'
    elif MinDigits < 0:
        T_Digits = str(MinDigits) + 'Digits'
    elif MinDigits == 0:
        T_Digits = 'AllDigits'
    T_NumBins = 'Bin' + str(NumBins)
    if ExtentName == 'Buenos Aires':
        T_ext = T_NumBins + 'CABA'
    elif ExtentName == 'Main Buenos Aires':
        T_ext = T_NumBins + 'MainBA'
    else:
        T_ext = T_NumBins + ExtentName
    if Absolute:
        T_abs = 'AbsScale'
    else:
        T_abs = 'RelScale'
    T_scale = T_abs + str(Scale)
    title = ','.join([KeyWord,T_UserType,T_TimeFrame,T_Digits,T_ext,T_scale])
    return title

def GetTitle2(TitleElts):
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale, BinType] = TitleElts
    T_UserType = UserType+'Users'
    T_TimeFrame = TimeFrame
    if MinDigits > 0:
        T_Digits = str(MinDigits) + '+Digits'
    elif MinDigits < 0:
        T_Digits = str(MinDigits) + 'Digits'
    elif MinDigits == 0:
        T_Digits = 'AllDigits'
    if BinType == 'Matrix':
        T_Bins = 'Bin' + str(NumBins)
        if ExtentName == 'Buenos Aires':
            T_ext = T_Bins + 'CABA'
        elif ExtentName == 'Main Buenos Aires':
            T_ext = T_Bins + 'MainBA'
        else:
            T_ext = T_Bins + ExtentName
    elif BinType == 'Coordinates':
        T_ext = 'BinCoord'
    if Absolute:
        T_abs = 'AbsScale'
    else:
        T_abs = 'RelScale'
    T_scale = T_abs + str(Scale)
    title = ','.join([KeyWord,T_UserType,T_TimeFrame,T_Digits,T_ext,T_scale])
    return title

def GetTitleBase(TitleElts):
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    T_UserType = UserType+'Users'
    T_TimeFrame = TimeFrame
    if MinDigits > 0:
        T_Digits = str(MinDigits) + '+Digits'
    elif MinDigits < 0:
        T_Digits = str(MinDigits) + 'Digits'
    elif MinDigits == 0:
        T_Digits = 'AllDigits'
    T_NumBins = 'Bin' + str(NumBins)
    if ExtentName == 'Buenos Aires':
        T_ext = T_NumBins + 'CABA'
    elif ExtentName == 'Main Buenos Aires':
        T_ext = T_NumBins + 'MainBA'
    else:
        T_ext = T_NumBins + ExtentName
    if Absolute:
        T_abs = 'AbsScale'
    else:
        T_abs = 'RelScale'
    T_scale = T_abs + str(Scale)
    title = ','.join([KeyWord,T_UserType,T_TimeFrame,T_Digits,T_ext,T_scale])
    TitleBase = ','.join([KeyWord,T_UserType,T_TimeFrame,T_Digits,T_ext])
    return TitleBase


def GetScale(ExtentName, Squares, Absolute, RelScale):
    if Squares:
        if ExtentName == 'Buenos Aires':
            Scale = 77
        elif ExtentName == 'Main Buenos Aires':
            Scale = 58
        elif ExtentName == 'Exact CABA':
            Scale = 64
        else:
            Scale = 68
    else:
        if Absolute:
            Scale = RelScale * 0.1
        else:
            Scale = RelScale * 10000
    return Scale
    #Scale = GetScale(ExtentName, Squares, Absolute, RelScale)

def PlotIt(Show, ID_pndata, TitleElts, Squares):
    title = GetTitle(TitleElts)
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    extent = GetExtent(ExtentName)
    if Squares:
        title = title + 'Squares'
        MarkerScale = Scale
        circ_scale = 0
        fig = pltposneg_square_old(Show, MarkerScale, NumBins, ID_pndata, extent, circ_scale, title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
    else:
        title = title + 'Circles'
        circ_scale = Scale
        fig = pltposneg(Show, ID_pndata, extent, circ_scale, title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
    return fig, datetimestr()+'_'+title+'.png'        
        
def PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha):
    title = GetTitle(TitleElts)
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    extent = GetExtent(ExtentName)
    if Squares:
        title = title + 'Squares'
        MarkerScale = Scale
        circ_scale = 0
        fig = pltposneg_square_old(Show, MarkerScale, NumBins, ID_pndata, extent, circ_scale, title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    else:
        title = title + 'Circles'
        circ_scale = Scale
        fig = pltposneg(Show, ID_pndata, extent, circ_scale, title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    return fig, title+'.png'        

def PlotIt_alpha_format(GraphFormat, Show, ID_pndata, TitleElts, Squares, alpha):
    title = GetTitle(TitleElts)
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    extent = GetExtent(ExtentName)
    if Squares:
        title = title + 'Squares'
        MarkerScale = Scale
        circ_scale = 0
        fig = pltposneg_square_format(GraphFormat, Show, MarkerScale, NumBins, ID_pndata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    else:
        title = title + 'Circles'
        circ_scale = Scale
        fig = pltposneg_format(GraphFormat, Show, ID_pndata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    return fig, title+'.png'        

def PlotIt_alpha_format_fill(GraphFormat, Show, ID_pndata, TitleElts, Squares, alpha):
    title = GetTitle(TitleElts)
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    extent = GetExtent(ExtentName)
    if Squares:
        title = title + 'Squares'
        fig = pltposneg_square_format_fill(GraphFormat, Show, NumBins, ID_pndata, extent, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    else:
        title = title + 'Circles'
        circ_scale = Scale
        fig = pltposneg_format(GraphFormat, Show, ID_pndata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    return fig, title+'.png'

def PlotIt_alpha_bdr(Show, ID_pndata, TitleElts, Squares, alpha, border):
    title = GetTitle(TitleElts)
    [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale] = TitleElts
    extent = GetExtent(ExtentName)
    if Squares:
        title = title + 'Squares'
        MarkerScale = Scale
        circ_scale = 0
        fig = pltposneg_square_old(Show, MarkerScale, NumBins, ID_pndata, extent, circ_scale, title, img='y', bdr=border, grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    else:
        title = title + 'Circles'
        circ_scale = Scale
        fig = pltposneg(Show, ID_pndata, extent, circ_scale, title, img='y', bdr=border, grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
    return fig, title+'.png'        

def CheckDataInExtent(GeoData, extent):
	MaxLon,MinLon = max(extent[0],extent[1]), min(extent[0],extent[1])
	MaxLat,MinLat = max(extent[2],extent[3]), min(extent[2],extent[3])
	InAll = []
	for i in range(len(GeoData[0])):
		In = MinLon <= GeoData[0][i] <= MaxLat and MinLat <= GeoData[1][i] <= MaxLat
		InAll.append(In)
	return all(InAll)

def GetNBins(extent, pndata):
	LonSet = sorted(list(set(pndata[0])))
	LatSet = sorted(list(set(pndata[1])))
	dlon = LonSet[1] - LonSet[0]
	dlat = LatSet[1] - LatSet[0]
	#NLonBins = round(abs((extent[1] - extent[0]))/dlon)
	#NLatBins = round(abs((extent[2] - extent[3]))/dlat)
	NLonBins = abs((extent[1] - extent[0]))/dlon
	NLatBins = abs((extent[2] - extent[3]))/dlat
	return NLonBins, NLatBins

def collocation(Text, NumOutputs):
##    import nltk
##    from nltk.tokenize import word_tokenize
##    from nltk.collocations import *
##    from string import *
##    from collections import defaultdict
##    from operator import itemgetter
##    from nltk.corpus import stopwords
##    from nltk.probability import FreqDist   
##    nltk.download("stopwords")
    ls=[]
    for line in Text:
        tok= word_tokenize(line)
        ls.append(tok)
    flat= [item for sublist in ls for item in sublist]
    stop_words= stopwords.words("spanish")
    stop_words_add= ["...", "El", "la", "La", "el", "A", "a","Repost","get_repost","・・・","(",")","``","lt","gt","--","-","•","{", "}",";", ":", "'", ",", "<", ">", ".","/","?","@","#","$","%", "^", "&", "*", "_",  "~",  ":", ",", "]", '[', '!', '/', "http", "https", '#', '://']
    new_words = [word for word in flat if word not in stop_words]
    new_new_words=[word for word in new_words if word not in stop_words_add]
    #print(new_new_words[0:100])
    #fdist= FreqDist(new_new_words)
    #print(fdist.most_common(100))
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    #trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(new_new_words)
    collocations = finder.nbest(bigram_measures.likelihood_ratio, NumOutputs)
    #print(collocations)
    return collocations

def FreqPlot(x,y,fit, avgline, staterr, title):
	sxy = sorted(zip(x,y))
	sx = [a for a,b in sxy]
	sy = [b for a,b in sxy]
	SumY = sum(sy)
	SumX = sum(sx)
	R = SumY/SumX
	plt.scatter(sx,sy,s=3, c = 'black')
	if fit == True:
		#m,b = np.polyfit(X,Y,1)
		#plt.plot(X, m*X+b, color = 'red')
		m,b = np.polyfit(sx,sy,1)
		plt.plot(sx, [m*x+b for x in sx], color = 'red')
	if avgline == True:
		plt.plot(sx, [x*R for x in sx], '--g')
	if staterr == True:
		plt.plot(sx, [(x+np.sqrt(x))*R for x in sx], color = 'grey')
		plt.plot(sx, [(x-np.sqrt(x))*R for x in sx], color = 'grey')
	m,b,r,p,stderr = linregress(x, y)
	StatsStr = 'mExact={:.4}, mFit{:.4}, y0={:.3}, rPear={:.3}, p={}, Err={:.4}'.format(R,m,b,r,p,stderr)
	plt.xlabel('Frequency of Reference Tweets')
	plt.ylabel('Frequency of Target Tweets')
	plt.title('Frequency Comparison_'+title+'\n' + StatsStr)
	plt.show()
	#FreqPlot(Matrix_ref_flat, Matrix_words_flat, fit=True, avgline=True, staterr=True, title='KMA')

def FreqPlot2(x,y,fit, avgline, staterr, title, FontSize, CircleSize):
	sxy = sorted(zip(x,y))
	sx = [a for a,b in sxy]
	sy = [b for a,b in sxy]
	SumY = sum(sy)
	SumX = sum(sx)
	R = SumY/SumX
	fig = plt.figure(figsize=(7,6))
	plt.scatter(sx,sy,s=CircleSize, facecolors='none', edgecolors='black')
	#plt.tight_layout()
	plt.gcf().subplots_adjust(bottom=.1, left=.2, top=1-.1, right=1-.1)
	if fit == True:
		m,b = np.polyfit(sx,sy,1)
		plt.plot(sx, [m*X+b for X in sx], color = 'red')
	if avgline == True:
		plt.plot(sx, [X*R for X in sx], '--g')
	if staterr == True:
		plt.plot(sx, [(X+np.sqrt(X))*R for X in sx], color = 'grey')
		plt.plot(sx, [(X-np.sqrt(X))*R for X in sx], color = 'grey')
	m,b,r,p,stderr = linregress(x, y)
	StatsStr = 'mExact{:.4}_mFit{:.4}_y0{:.3}_rPear{:.4f}_p{:.4f}_StdErr{:.4f}'.format(R,m,b,r,p,stderr)
	plt.xlabel('Reference Tweet Counts', fontsize = FontSize)
	plt.ylabel('Target Tweets Counts', fontsize = FontSize)
	matplotlib.rc('xtick', labelsize=FontSize)
	matplotlib.rc('ytick', labelsize=FontSize)
	plt.title(title+'\n' + StatsStr)
	plt.show()
	return fig, linregress(x, y)
	#FreqPlot2(Matrix_ref_flat, Matrix_words_flat, fit=True, avgline=True, staterr=True, title='KMA')


def AngleHist(x,y, title, avgline):
    sxy = sorted(zip(x,y))
    sx = [a for a,b in sxy]
    sy = [b for a,b in sxy]
    X = np.array(sx)
    Y = np.array(sy)
    r = sum(Y)/sum(X)
    A0 = np.arctan(r)*180/np.pi
    Angles = []
    X_subset = []
    for i in range(len(X)):
        if X[i] == 0 and Y[i]> 0:
            Angles.append(90)
            X_subset.append(X[i])
        if X[i]>0:
            Angles.append(np.arctan(Y[i]/X[i])*180/np.pi)
            X_subset.append(X[i])
    #Angles = np.arctan(Y/X)*180/np.pi
    stddev = np.std(Angles)
    AngleBins = [x for x in range(91)]
    AngleBinCenters = [x + 0.5 for x in range(90)]
    Hist = [0 for x in range(90)]
    for i in range(len(Angles)):
        ind = np.searchsorted(AngleBins, Angles[i], side="left") - 1
        Hist[int(ind)] += X_subset[i]
        #Hist[int(ind)] += 1
    #plt.scatter(Angles, X_subset,s=3, c = 'black')
    plt.scatter(AngleBinCenters,Hist, s=3, c = 'black')
    if avgline == True:
        plt.plot([A0,A0],[0,max(Hist)], '--g')
    plt.xlabel('Angle [deg]')
    plt.ylabel('Number of Reference Tweets')
    plt.title('Angle Histogram: ' + 'sigma=' + str(stddev)[0:5] + '_' + title)
    plt.show()
    return stddev
    #AngleHist(Matrix_ref_flat, Matrix_words_flat, graphname = 'KMA', avgline=True)

def AngleHist_Norm(x,y, title, avgline):
    sxy = sorted(zip(x,y))
    sx = [a for a,b in sxy]
    sy = [b for a,b in sxy]
    x = np.array(sx)
    y = np.array(sy)
    Sumx = sum(x)
    Sumy = sum(y)
    X = x/Sumx
    Y = y/Sumy
    r = Sumy/Sumx
    R = sum(Y)/sum(X)
    a0 = np.arctan(r)*180/np.pi
    A0 = np.arctan(R)*180/np.pi
    Angles = []
    X_subset = []
    for i in range(len(X)):
        if X[i] == 0 and Y[i]> 0:
            Angles.append(90)
            X_subset.append(X[i])
        if X[i]>0:
            Angles.append(np.arctan(Y[i]/X[i])*180/np.pi)
            X_subset.append(X[i])
    #Angles = np.arctan(Y/X)*180/np.pi
    stddev = np.std(Angles)
    AngleBins = [x for x in range(91)]
    AngleBinCenters = [x + 0.5 for x in range(90)]
    Hist = [0 for x in range(90)]
    for i in range(len(Angles)):
        ind = np.searchsorted(AngleBins, Angles[i], side="left") - 1
        Hist[int(ind)] += X_subset[i]
        #Hist[int(ind)] += 1
    #plt.scatter(Angles, X_subset,s=3, c = 'black')
    plt.scatter(AngleBinCenters,Hist, s=3, c = 'black')
    if avgline == True:
        plt.plot([A0,A0],[0,max(Hist)], '--g')
    plt.xlabel('Angle [deg]')
    plt.ylabel('Fraction of Reference Tweets in Bin')
    plt.title('Normalized Angle Histogram: ' + 'sigma=' + str(stddev)[0:5] + '_' + title)
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    plt.show()
    return stddev
    #AngleHist_Norm(Matrix_ref_flat, Matrix_words_flat, title = 'KMA', avgline=True)

def AngleHist_Norm2(x,y, title, avgline, FontSize, CircleSize):
	sxy = sorted(zip(x,y))
	sx, sy = [a for a,b in sxy], [b for a,b in sxy]
	x,y = np.array(sx), np.array(sy)
	Sumx, Sumy = sum(x), sum(y)
	X,Y = x/Sumx, y/Sumy
	r,R = Sumy/Sumx, sum(Y)/sum(X)
	a0, A0 = np.arctan(r)*180/np.pi, np.arctan(R)*180/np.pi
	Angles, X_subset = [], []
	for i in range(len(X)):
		if X[i] == 0 and Y[i]> 0:
			Angles.append(90)
			X_subset.append(X[i])
		if X[i]>0:
			Angles.append(np.arctan(Y[i]/X[i])*180/np.pi)
			X_subset.append(X[i])
	stddev = np.std(Angles)
	AngleBins = [x for x in range(91)]
	AngleBinCenters = [x + 0.5 for x in range(90)]
	Hist = [0 for x in range(90)]
	for i in range(len(Angles)):
		ind = np.searchsorted(AngleBins, Angles[i], side="left") - 1
		Hist[int(ind)] += X_subset[i]
	fig = plt.figure(figsize=(7,6))
	plt.gcf().subplots_adjust(bottom=.12, left=.17, top=1-.1, right=1-.08)
	plt.scatter(AngleBinCenters,Hist, s=CircleSize, facecolors='none', edgecolors='black')
	if avgline == True:
		plt.plot([A0,A0],[0,max(Hist)], '--g')
	plt.xlabel('Normalized Angle [deg]', fontsize = FontSize)
	plt.ylabel('Fraction of Reference Tweets per degree', fontsize = FontSize)
	plt.title(title + '\n' + 'NormAngleHist: ' + 'sigma=' + str(stddev)[0:5])
	plt.rc('xtick', labelsize=FontSize)
	plt.rc('ytick', labelsize=FontSize)
	plt.show()
	return fig, stddev
	#AngleHist_Norm(Matrix_ref_flat, Matrix_words_flat, title = 'KMA', avgline=True)

def AngleHist_Norm2(x,y, title, avgline, FontSize, CircleSize):
	sxy = sorted(zip(x,y))
	sx, sy = [a for a,b in sxy], [b for a,b in sxy]
	x,y = np.array(sx), np.array(sy)
	Sumx, Sumy = sum(x), sum(y)
	X,Y = x/Sumx, y/Sumy
	r,R = Sumy/Sumx, sum(Y)/sum(X)
	a0, A0 = np.arctan(r)*180/np.pi, np.arctan(R)*180/np.pi
	Angles, X_subset = [], []
	for i in range(len(X)):
		if X[i] == 0 and Y[i]> 0:
			Angles.append(90)
			X_subset.append(X[i])
		if X[i]>0:
			Angles.append(np.arctan(Y[i]/X[i])*180/np.pi)
			X_subset.append(X[i])
	stddev = np.std(Angles)
	AngleBins = [x for x in range(91)]
	AngleBinCenters = [x + 0.5 for x in range(90)]
	Hist = [0 for x in range(90)]
	for i in range(len(Angles)):
		ind = np.searchsorted(AngleBins, Angles[i], side="left") - 1
		Hist[int(ind)] += X_subset[i]
	fig = plt.figure(figsize=(7,6))
	plt.gcf().subplots_adjust(bottom=.12, left=.17, top=1-.1, right=1-.08)
	plt.scatter(AngleBinCenters,Hist, s=CircleSize, facecolors='none', edgecolors='black')
	if avgline == True:
		plt.plot([A0,A0],[0,max(Hist)], '--g')
	plt.xlabel('Normalized Angle [deg]', fontsize = FontSize)
	plt.ylabel('Fraction of Reference Tweets per degree', fontsize = FontSize)
	plt.title(title + '\n' + 'NormAngleHist: ' + 'sigma=' + str(stddev)[0:5])
	plt.rc('xtick', labelsize=FontSize)
	plt.rc('ytick', labelsize=FontSize)
	current_values = plt.gca().get_yticks()
	plt.gca().set_yticklabels(['{:.2f}'.format(x) for x in current_values])
	plt.show()
	return fig, stddev
	#AngleHist_Norm(Matrix_ref_flat, Matrix_words_flat, title = 'KMA', avgline=True)

def AngleHist_Norm_NonZero(x,y, title, avgline, FontSize, CircleSize):
	sxy = sorted(zip(x,y))
	sx, sy = [a for a,b in sxy], [b for a,b in sxy]
	x,y = np.array(sx), np.array(sy)
	Sumx, Sumy = sum(x), sum(y)
	X,Y = x/Sumx, y/Sumy
	r,R = Sumy/Sumx, sum(Y)/sum(X)
	a0, A0 = np.arctan(r)*180/np.pi, np.arctan(R)*180/np.pi
	Angles, X_subset = [], []
	for i in range(len(X)):
		if X[i] == 0 and Y[i]> 0:
			Angles.append(90)
			X_subset.append(X[i])
		if X[i]>0:
			Angles.append(np.arctan(Y[i]/X[i])*180/np.pi)
			X_subset.append(X[i])
	stddev = np.std(Angles)
	AngleBins = [x for x in range(91)]
	AngleBinCenters = [x + 0.5 for x in range(90)]
	Hist = [0 for x in range(90)]
	for i in range(len(Angles)):
		ind = np.searchsorted(AngleBins, Angles[i], side="left") - 1
		Hist[int(ind)] += X_subset[i]
	if True: #plot only non-zero histogram bins
		Non0 = [[Hist[i], AngleBinCenters[i]] for i in range(len(Hist)) if Hist[i] != 0]
		Hist = [x[0] for x in Non0]
		AngleBinCenters = [x[1] for x in Non0]
	fig = plt.figure(figsize=(7,6))
	plt.gcf().subplots_adjust(bottom=.12, left=.17, top=1-.1, right=1-.08)
	plt.scatter(AngleBinCenters,Hist, s=CircleSize, facecolors='none', edgecolors='black')
	if avgline == True:
		plt.plot([A0,A0],[0,max(Hist)], '--g')
	plt.xlabel('Normalized Angle [deg]', fontsize = FontSize)
	plt.ylabel('Fraction of Reference Tweets per degree', fontsize = FontSize)
	plt.title(title + '\n' + 'NormAngleHist: ' + 'sigma=' + str(stddev)[0:5])
	plt.rc('xtick', labelsize=FontSize)
	plt.rc('ytick', labelsize=FontSize)
	plt.show()
	return fig, stddev
	#AngleHist_Norm(Matrix_ref_flat, Matrix_words_flat, title = 'KMA', avgline=True)

def mkscatplt2(coord, freq, extent, circ_scale, title, img, bdr, grd, clr, cmap, alpha, osm_img):
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    lons = coord[0]
    lats = coord[1]
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        circlesizes = [-circ_scale for x in freq]
    else:
        circlesizes = [abs(x)*circ_scale for x in freq]
    if clr == 'scale':
        plt.scatter(x=lons, y=lats, c=freq, s=circlesizes, alpha=alpha, cmap=cmap, transform=crs.PlateCarree())
    else:
        plt.scatter(x=lons, y=lats, color=clr, s=circlesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.show()
    return fig
    #mkscatplt2(coord, freq, extent, circ_scale = 0.02, title = 'KMA', img='y', bdr='y', grd='n', clr='purple', cmap='bwr', alpha = 0.3, osm_img = osm_img)

def mkscatplt_show(show, coord, freq, extent, circ_scale, title, img, bdr, grd, clr, cmap, alpha, osm_img, fontsize):
    #fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    fig = plt.figure() # open matplotlib figure without specifying the size
    TitleFontSize, AxesFontSize = fontsize, fontsize
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=TitleFontSize)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
    ax1.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
    ax1.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    lons = coord[0]
    lats = coord[1]
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        circlesizes = [-circ_scale for x in freq]
    else:
        circlesizes = [abs(x)*circ_scale for x in freq]
    if clr == 'scale':
        plt.scatter(x=lons, y=lats, c=freq, s=circlesizes, alpha=alpha, cmap=cmap, transform=crs.PlateCarree())
    else:
        plt.scatter(x=lons, y=lats, color=clr, s=circlesizes, alpha=alpha,transform=crs.PlateCarree())
    plt.tight_layout()
    if show:
        plt.show()
    return fig
    #mkscatplt_show(show, coord, freq, extent, circ_scale = 0.02, title = 'KMA', img='y', bdr='y', grd='n', clr='purple', cmap='bwr', alpha = 0.3, osm_img = osm_img)

def mkscatplt_show_format(GraphFormat, show, coord, freq, extent, circ_scale, title, img, bdr, grd, clr, cmap, alpha, osm_img):
	if GraphFormat['FigSize'] == []:
		fig = plt.figure() # open matplotlib figure
	else:
		fig = plt.figure(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1])) # open matplotlib figure
	TitleFontSize, AxesFontSize = GraphFormat['TitleFontSize'], GraphFormat['AxesFontSize']
	xtics,ytics = GraphFormat['NumTics']
	ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
	ax1.set_title(title,fontsize=TitleFontSize)
	if bdr == 'y':
		ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
		ax1.add_feature(cfeature.BORDERS, edgecolor="black")
	if grd == 'y':
		ax1.gridlines()
	ax1.set_extent(extent) # set extents
	ax1.set_xticks(np.linspace(extent[0],extent[1],xtics),crs=ccrs.PlateCarree()) # set longitude indicators
	ax1.set_yticks(np.linspace(extent[2],extent[3],ytics)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
	lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
	lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
	ax1.xaxis.set_major_formatter(lon_formatter) # set lons
	ax1.yaxis.set_major_formatter(lat_formatter) # set lats
	ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
	ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
	ax1.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
	ax1.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
	ax1.set_title(title)
	scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
	scale = (scale<20) and scale or 19 # scale cannot be larger than 19
	lons = coord[0]
	lats = coord[1]
	if img == 'y':
		ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
	if circ_scale < 0:
		circlesizes = [-circ_scale for x in freq]
	else:
		circlesizes = [abs(x)*circ_scale for x in freq]
	if clr == 'scale':
		plt.scatter(x=lons, y=lats, c=freq, s=circlesizes, alpha=alpha, cmap=cmap, transform=crs.PlateCarree())
	else:
		plt.scatter(x=lons, y=lats, color=clr, s=circlesizes, alpha=alpha,transform=crs.PlateCarree())
	if GraphFormat['tight']:
		plt.tight_layout()
	if show:
		plt.show()
	return fig
	#mkscatplt_show(show, coord, freq, extent, circ_scale = 0.02, title = 'KMA', img='y', bdr='y', grd='n', clr='purple', cmap='bwr', alpha = 0.3, osm_img = osm_img)

    def W(i, j, NBins):
    row_i = (i - i%NBins)/NBins
    row_j = (j - j%NBins)/NBins
    col_i = i%NBins
    col_j = j%NBins
    d_row = abs(row_i - row_j)
    d_col = abs(col_i - col_j)
    if (d_row == 1 and d_col<=1) or (d_col == 1 and d_row <= 1):
        w = 1
    else:
        w=0
    return w

def SquareNeighborIndices(index,NBins,Layers):
    index_list = []
    for i in range(-Layers,Layers+1,1):
        for j in range(-Layers,Layers+1,1):
            if i != 0 or j != 0:
                NeighborIndex = int(index + i*NBins+j)
                if 0 <= NeighborIndex < NBins**2:
                    index_list.append(NeighborIndex)
    return index_list

def EdgeNeighborIndices(index,NBins):
	index_list = []
	for x in [-NBins,-1,1,NBins]:
		NeighborIndex = int(index + x)
		if 0 <= NeighborIndex < NBins**2:
		    index_list.append(NeighborIndex)
	return index_list
    
def CheckeredData(NBins,Size):
	Values1D = []
	Values2D = []
	N = NBins**2
	for col in range(NBins):
		RowValues = []
		for row in range(NBins):
			value = (math.floor(row/Size) + math.floor(col/Size))%2
			Values1D.append(value)
			RowValues.append(value)
		Values2D.append(RowValues)
	return Values1D, Values2D

def CircleNeighborIndices(index,NBins,Layers):
    index_list = []
    for i in range(-Layers,Layers+1,1):
        for j in range(-Layers,Layers+1,1):
            if (i != 0 or j != 0) and np.sqrt(i**2+j**2) <= int(Layers):
                NeighborIndex = int(index + i*NBins+j)
                if 0 <= NeighborIndex < NBins**2:
                    index_list.append(NeighborIndex)
    return index_list

def CircleNeighborIndices_Non0(index,NBins,Layers, Counts):
    index_list = []
    for i in range(-Layers,Layers+1,1):
        for j in range(-Layers,Layers+1,1):
            if (i != 0 or j != 0) and np.sqrt(i**2+j**2) <= int(Layers):
                NeighborIndex = int(index + i*NBins+j)
                if 0 <= NeighborIndex < NBins**2:
                    if Counts[NeighborIndex] != 0:
                        index_list.append(NeighborIndex)
    return index_list

def CircleLayerIndices(index,NBins,Layers):
	IndicesByLayer = [[] for i in range(Layers)]
	for i in range(-Layers,Layers+1,1):
		for j in range(-Layers,Layers+1,1):
			dInd = np.sqrt(i**2+j**2)
			L = math.ceil(dInd)
			#print('i: {}, j: {}, L: {}'.format(i,j,L))
			NeighborIndex = int(index + i*NBins+j)
			if (i != 0 or j != 0) and (L <= Layers) and (0 <= NeighborIndex < NBins**2):
				IndicesByLayer[L-1].append(NeighborIndex)
	return IndicesByLayer

def CircleLayerIndices_Non0(index,NBins,Layers, Counts):
	IndicesByLayer = [[] for i in range(Layers)]
	if Counts[index] != 0:
		for i in range(-Layers,Layers+1,1):
			for j in range(-Layers,Layers+1,1):
				dInd = np.sqrt(i**2+j**2)
				L = math.ceil(dInd)
				#print('i: {}, j: {}, L: {}'.format(i,j,L))
				NeighborIndex = int(index + i*NBins+j)
				if (i != 0 or j != 0) and (L <= Layers) and (0 <= NeighborIndex < NBins**2):
					if Counts[NeighborIndex] != 0:
						IndicesByLayer[L-1].append(NeighborIndex)
	return IndicesByLayer

def NeighborIndices(index,NBins, Layers, NeighborType):
    if NeighborType == "Edge":
        index_list = EdgeNeighborIndices(index,NBins)
    elif NeighborType == "Square":
        index_list = SquareNeighborIndices(index,NBins,Layers)
    elif NeighborType == "Circle":
        index_list = CircleNeighborIndices(index, NBins, Layers)
    return index_list


def AvgNeighbors(N):
	if N == 1:
		AvgValue = -1
	else:
		Nc = 4
		Ne = (N-1)*4 - Nc
		Nm = N*N - Ne - Nc
		AvgValue = 2*((Nc*0.5 + Ne*0.75 + Nm)/(Nc + Ne + Nm)) - 1
	return AvgValue

if False:
    MI_Check = [MoranI(CheckeredData(100,i)[0],1)[0] for i in range(1,21,1)]
    x_val = [i for i in range(1,21,1)]
    plt.plot(x_val,MI_Check, c = 'black')
    plt.scatter(x_val,[AvgNeighbors(N) for N in x_val], s=20, c = 'red')
    plt.xlabel('Cluster Size [N x N]')
    plt.ylabel('Moran\'s I')
    plt.title('Value of Moran\'s I vs. Cluster Size')
    plt.show()

##def MoranI(Counts,Layers):
##    N = len(Counts)
##    x = Counts
##    xbar = np.mean(x)
##    NBins = np.sqrt(N)
##    OneSum = sum([(x[i]-xbar)**2 for i in range(N)])
##    wSum = sum([len(NeighborIndices(i,NBins,Layers)) for i in range(N)])
##    DSum = sum([sum([(x[i]-xbar)*(x[j]-xbar) for j in NeighborIndices(i,NBins,Layers)]) for i in range(N)])
##    moranI = N * DSum/(OneSum*wSum)
##    EI = -1/(N-1)
##    S1 = 2*wSum
##    S2 = 4*sum([(len(NeighborIndices(i,NBins,Layers)))**2 for i in range(N)])
##    S3n = sum([(x[i]-xbar)**4 for i in range(len(x))])/N
##    S3d = ( sum([(x[i]-xbar)**2 for i in range(len(x))])/N )**2
##    S3 = S3n/S3d
##    S4 = (N**2 - 3*N +3)*S1 - N*S2 + 3*wSum**2
##    S5 = S1 - 2*N*S1 + 6*wSum**2
##    varI = (N*S4 - S3*S5)/( (N-1)*(N-2)*(N-3)*wSum**2 )-EI**2
##    z = (moranI - EI)/(varI)**0.5
##    description = 'Moran\'s I, E(I), Var(I), z'
##    return moranI, EI, varI, z, description

def MoranI(Counts,Layers,Type):
	N = len(Counts)
	x = Counts
	xbar = np.mean(x)
	NBins = np.sqrt(N)
	OneSum = sum([(x[i]-xbar)**2 for i in range(N)])
	wSum = sum([len(NeighborIndices(i,NBins,Layers,Type)) for i in range(N)])
	DSum = sum([sum([(x[i]-xbar)*(x[j]-xbar) for j in NeighborIndices(i,NBins,Layers,Type)]) for i in range(N)])
	moranI = N * DSum/(OneSum*wSum)
	EI = -1/(N-1)
	S1 = 2*wSum
	S2 = 4*sum([(len(NeighborIndices(i,NBins,Layers,Type)))**2 for i in range(N)])
	S3n = sum([(x[i]-xbar)**4 for i in range(len(x))])/N
	S3d = ( sum([(x[i]-xbar)**2 for i in range(len(x))])/N )**2
	S3 = S3n/S3d
	S4 = (N**2 - 3*N +3)*S1 - N*S2 + 3*wSum**2
	S5 = S1 - 2*N*S1 + 6*wSum**2
	varI = (N*S4 - S3*S5)/( (N-1)*(N-2)*(N-3)*wSum**2 )-EI**2
	z = (moranI - EI)/(varI)**0.5
	description = 'Moran\'s I, E(I), Var(I), z'
	return moranI, EI, varI, z, description

def L_MoranI(Counts,Layers):
	N = len(Counts)
	x = Counts
	xbar = np.mean(x)
	NBins = np.sqrt(N)
	OneSum = sum([(x[i]-xbar)**2 for i in range(N)])
	L_wSum = [sum([len(CircleLayerIndices(i,NBins,Layers)[L]) for i in range(N)]) for L in range(Layers)]
	L_DSum = [sum([sum([(x[i]-xbar)*(x[j]-xbar) for j in CircleLayerIndices(i,NBins,Layers)[L]]) for i in range(N)]) for L in range(Layers)]
	L_moranI = [N * sum(L_DSum[0:L+1])/(OneSum*sum(L_wSum[0:L+1])) for L in range(Layers)]
	EI = -1/(N-1)
	L_S1 = [2*sum(L_wSum[L]) for L in range(Layers)]
	L_S2 = [4*sum([(len(CircleLayerIndices(i,NBins,Layers)[L]))**2 for i in range(N)]) for L in range(Layers)]
	S3n = sum([(x[i]-xbar)**4 for i in range(len(x))])/N
	S3d = ( sum([(x[i]-xbar)**2 for i in range(len(x))])/N )**2
	S3 = S3n/S3d
	L_S4 = [(N**2 - 3*N +3)*sum(L_S1[0:L+1]) - N*sum(L_S2[0:L+1]) + 3*sum(L_wSum[0:L+1])**2 for L in range(Layers)]
	L_S5 = [(1 - 2*N)*sum(L_S1[0:L+1]) + 6*sum(L_wSum[0:L+1])**2 for L in range(Layers)]
	L_varI = [(N*sum(L_S4[0:L+1]) - S3*sum(L_S5[0:L+1]))/( (N-1)*(N-2)*(N-3)*sum(L_wSum[0:L+1])**2 )-EI**2 for L in range(Layers)]
	L_z = [(sum(L_moranI[0:L+1]) - EI)/(sum(L_varI[0:L+1]))**0.5 for L in range(Layers)]
	description = 'Moran\'s I, E(I), Var(I), z in arrays, 1 element per layer'
	return L_moranI, EI, L_varI, L_z, description

def MoranI_NonZero(Counts,Layers):
	N = len(Counts)
	NBins = np.sqrt(N)
	Non0_Inds = [i for i in range(len(Counts)) if Counts[i] !=0]
	x_Non0 = [x for x in Counts if x!=0]
	N_Non0 = len(x_Non0)
	xbar_Non0 = np.mean(x_Non0)
	OneSum_Non0 = sum([(x_Non0[i]-xbar_Non0)**2 for i in range(N_Non0)])
	wSum_Non0 = sum([len(CircleNeighborIndices_Non0(i,NBins,Layers, Counts)) for i in Non0_Inds])
	DSum_Non0 = sum([sum([(Counts[i]-xbar_Non0)*(Counts[j]-xbar_Non0) for j in CircleNeighborIndices_Non0(i,NBins,Layers, Counts)]) for i in Non0_Inds])
	moranI = N_Non0 * DSum_Non0/(OneSum_Non0*wSum_Non0)
	EI = -1/(N_Non0-1)
	S1 = 2*wSum_Non0
	S2 = 4*sum([(len(CircleNeighborIndices_Non0(i,NBins,Layers, Counts)))**2 for i in range(N_Non0)])
	S3n = sum([(Counts[i]-xbar_Non0)**4 for i in Non0_Inds])/N_Non0
	S3d = ( sum([(Counts[i]-xbar_Non0)**2 for i in Non0_Inds])/N_Non0 )**2
	S3 = S3n/S3d
	S4 = (N_Non0**2 - 3*N_Non0 +3)*S1 - N_Non0*S2 + 3*wSum_Non0**2
	S5 = S1 - 2*N_Non0*S1 + 6*wSum_Non0**2
	varI = (N_Non0*S4 - S3*S5)/( (N_Non0-1)*(N_Non0-2)*(N_Non0-3)*wSum_Non0**2 )-EI**2
	z = (moranI - EI)/(varI)**0.5
	description = 'Moran\'s I, E(I), Var(I), z'
	return moranI, EI, varI, z, description

def WriteMoranIStatsFile(filepath, MI_Stats):
	f = open(filepath, "w")
	print('# of Layers\tMoran\'s I\tE(I)\tVar(I)\tz', file = f)
	for i in range(len(MI_Stats)):
		print('{}\t{}\t{}\t{}\t{}'.format(i+1,MI_Stats[i][0], MI_Stats[i][1], MI_Stats[i][2], MI_Stats[i][3]), file = f)
	f.close()

def LoadMoranIData(DataPath):
    print('Loading Moran\'s I data from file...')
    with open(DataPath, 'r', encoding='utf-8') as ff:
        ListLines, ListBadLines =[], []
        header = ff.readline()
        NumLayers, MoranI = [],[]
        count, count_err = 0, 0
        for line in ff:
            try:
                ListLines.append(line)
                count += 1
                LineElts = line.strip().split('\t')
                NumLayers.append(float(LineElts[0]))
                MoranI.append(float(LineElts[1]))
                if count%10000 == 0:
                    print('Lines Loaded: {}'.format(count))
            except:
                ListBadLines.append(line)
                count_err += 1
                if count_err%100 == 0:
                    print('Line parsing failed. Number of failed lines: {}'.format(count_err))
    Data = [NumLayers, MoranI]
    print(header.strip())
    print('{}\t{}'.format(NumLayers[0],MoranI[0]))
    print('{}\t{}'.format(NumLayers[1],MoranI[1]))
    return Data #, ListLines, ListBadLines

def KolmogorovSmirnov(Data, NumKSBins):
	R, T = Data[2], Data[3]
	S_R, S_T = sum(R), sum(T)
	if True: #equalize the number statistics
		if S_T > S_R:
			T = [int(x*S_R/S_T) for x in T]
		if S_T < S_R:
			R = [int(x*S_T/S_R) for x in R]
	S_R, S_T = sum(R), sum(T)
	f_R, f_T = [r/S_R for r in R], [t/S_T for t in T]
	N_R, N_T = len([x for x in R if x!=0]), len([x for x in T if x!=0])
	KS_N = NumKSBins
	KS_min = min([min(f_R), min(f_T)])
	#KS_max = max([max(f_R), max(f_T)])
	KS_max = 0.001
	KS_delta = (KS_max - KS_min)/(KS_N - 1)
	KS_x = [KS_min + i*KS_delta for i in range(KS_N)]
	KS_y_R = [len([f for f in f_R if f <= x])/len(R) for x in KS_x]
	KS_y_T = [len([f for f in f_T if f <= x])/len(T) for x in KS_x]
	plt.plot(KS_x, KS_y_R, c = 'Black')
	plt.plot(KS_x, KS_y_T, c = 'Red')
	plt.xlabel('Kolmogorov-Smirnov x')
	plt.ylabel('Cummulative Probability')
	plt.title('2-Sample Kolmogorov-Smirnov Test')
	plt.show()

def TwoSample2DKolmogorovSmirnov(Data1, Data2, Equalize,Plot):
	R, T = Data1, Data2
	N = len(R)
	S_R, S_T = sum(R), sum(T)
	if Equalize: #equalize the number statistics
		if S_T > S_R:
			T = [int(x*S_R/S_T) for x in T]
		if S_T < S_R:
			R = [int(x*S_T/S_R) for x in R]
	S_R, S_T = sum(R), sum(T)
	f_R, f_T = [r/S_R for r in R], [t/S_T for t in T]
	KS_x = [i for i in range(N)]
	KS_y_R = [sum(f_R[0:i]) for i in range(N)]
	KS_y_T = [sum(f_T[0:i]) for i in range(N)]
	Diff = [abs(KS_y_T[i] - KS_y_R[i]) for i in range(len(KS_y_T))]
	MaxDiff = max(Diff)
	P_Value = P_KS(MaxDiff,S_R,S_T,100)[0]
	if Plot:
		plt.plot(KS_x, KS_y_R, c = 'Black')
		plt.plot(KS_x, KS_y_T, c = 'Red')
		plt.xlabel('Kolmogorov-Smirnov x')
		plt.ylabel('Cummulative Probability')
		plt.title('2-Sample Kolmogorov-Smirnov Test')
		plt.show()
		plt.plot(KS_x, Diff)
		plt.xlabel('Kolmogorov-Smirnov x')
		plt.ylabel('Difference in Cummulative Probability')
		plt.title('Difference Test')
		plt.show()
	return MaxDiff, P_Value

def KSTest_Gaussian(x1,x2,dx1,dx2,NumTests):
	y1_Tests = []
	y2_Tests = []
	for i in range(NumTests):
		x = random.uniform(0,1)
		y1 = np.exp(-(x-x1)**2/dx1**2)
		y1_Tests.append(y1)
		x = random.uniform(0,1)
		y2 = np.exp(-(x-x2)**2/dx2**2)
		y2_Tests.append(y2)
	KSG_N = 200
	KSG_min = min([min(y1_Tests), min(y2_Tests)])
	KS_max = max([max(y1_Tests), max(y2_Tests)])
	#KSG_max = 0.001
	KSG_delta = (KS_max - KS_min)/(KS_N - 1)
	KSG_x = [KSG_min + i*KSG_delta for i in range(KSG_N)]
	KSG_y1 = [len([f for f in y1_Tests if f <= x])/len(y1_Tests) for x in KSG_x]
	KSG_y2 = [len([f for f in y2_Tests if f <= x])/len(y2_Tests) for x in KSG_x]
	plt.plot(KSG_x, KSG_y1, c = 'Black')
	plt.plot(KSG_x, KSG_y2, '--', c = 'Red')
	plt.xlabel('Kolmogorov-Smirnov x')
	plt.ylabel('Cummulative Probability')
	plt.title('2-Sample Kolmogorov-Smirnov Test')
	plt.show()

def TwoSampleKS(Data1, Data2,NumSamples1,NumSamples2):
	R, T = Data1, Data2
	MaxR, MinR = max(R),min(R)
	MaxT,MinT = max(T),min(T)
	Max = max([MaxR,MaxT])
	Min = min([MinR,MinT])
	N = max([NumSamples1,NumSamples2])
	dx = (Max - Min)/(N - 1)
	KS_x = [Min + i*dx for i in range(N)]
	KS_y_R = [len([x for x in R if x <= X])/len(R) for X in KS_x]
	KS_y_T = [len([x for x in T if x <= X])/len(T) for X in KS_x]
	Diff = [abs(KS_y_T[i] - KS_y_R[i]) for i in range(len(KS_y_T))]
	MaxDiff = max(Diff)
	pValue = P_KS(MaxDiff,NumSamples1,NumSamples2,100)[0]
	plt.plot(KS_x, KS_y_R, c = 'Black')
	plt.plot(KS_x, KS_y_T, c = 'Red')
	plt.xlabel('Kolmogorov-Smirnov x')
	plt.ylabel('Cummulative Probability')
	plt.title('2-Sample Kolmogorov-Smirnov Test')
	plt.show()
	plt.plot(KS_x, Diff)
	plt.xlabel('Kolmogorov-Smirnov x')
	plt.ylabel('Difference in Cummulative Probability')
	plt.title('Difference Test')
	plt.show()
	return MaxDiff, pValue

if False:
	from scipy import stats
	rng = np.random.default_rng()
	n1 = 200  # size of first sample
	n2 = 300  # size of second sample
	rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=rng)
	rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5, random_state=rng)
	stats.ks_2samp(rvs1, rvs2)
	TwoSampleKS(rvs1, rvs2, 200,300)
	rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0, random_state=rng)
	stats.ks_2samp(rvs1, rvs3)
	TwoSampleKS(rvs1, rvs3, 200,300)

def P_KS(D,n1,n2,NumIter):
	z = D*np.sqrt(n1*n2/(n1+n2))
	#reference: https://stats.stackexchange.com/questions/389034/kolmogorov-smirnov-test-calculating-the-p-value-manually
	z = D*np.sqrt(n1*n2/(n1+n2))
	p = 2*sum([(-1)**(i-1)*np.exp(-2*i**2*z**2) for i in range(1,NumIter+1,1)])
	return p,1-p, z

def alpha(D,n1,n2):
        #reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
	z = D*np.sqrt(n1*n2/(n1+n2))
	alpha = np.exp(-2*(z**2))
	return alpha, z

def c_alpha(alpha):
        #reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
	c = np.sqrt(-0.5*np.log(alpha/2))
	return c

if False:
	NumSamples = 1000
	r1 = norm.rvs(loc = 0.5, scale = 0.1, size=NumSamples)
	NumSteps = 500
	dloc = 0.1/(NumSteps - 1)
	S,P, Alpha = [],[], []
	for i in range(NumSteps):
		r2 = norm.rvs(loc = 0.5, scale = 0.1, size=NumSamples)
		s,p = ks_2samp(r1, r2)
		S.append(s)
		P.append(p)
		Alpha.append(alpha(s, NumSamples, NumSamples)[0])
	R = [P[i]/Alpha[i] for i in range(len(P))]
	k = [S[i]*np.sqrt(NumSamples/2) for i in range(len(S))]
	Z = zip(S,P,Alpha,R,k)
	tuples = zip(*sorted(Z))
	S_,P_,Alpha_,R_,k_ =[ list(tuple) for tuple in  tuples]
	K = [3/999*i for i in range(999)]
	NumIter = 10
	K_pdf = [2*sum([(-1)**(i-1)*np.exp(-2*(i**2)*(z**2)) for i in range(1,NumIter+1,1)]) for z in K]
	plt.scatter(k_,P_, s=20, c='black')
	plt.plot(k_,Alpha_, c='red')
	plt.plot(K,K_pdf, c='green')
	plt.xlabel('Kolmogorov-Smirnov Statistic x sqrt(n1*n2/n1+n2)')
	plt.ylabel('p-value')
	plt.title('Functional relationship of KS-Statistic to p-value')
	plt.show()

def GetMultiStats(Reference,Target, MI_NumLayers, MI_NonZero, KS_Equalize):
	R,T = Reference, Target
	if True: #Perform Regression Analysis
		#m,b,r,p,stderr = linregress(R,T)
		LinRegressStats = linregress(R,T)
	if True: #Perform Multi-layer Moran's I Analysis on Differential Distribution
		RBar, TBar = np.mean(R), np.mean(T)
		N = len(R)
		D_values = [T[i]/TBar - R[i]/RBar for i in range(N)]
		Layers = MI_NumLayers
		MI_Stats = []
		for L in range(1,Layers+1,1):
			t_0 = datetime.now()
			print('Layer number {}'.format(L))
			if MI_NonZero:
				MI = MoranI_NonZero(D_values,L)
			else:
				MI = MoranI(D_values, L, "Circle")
			MI_Stats.append(MI)
			print(MI)
			t_1 = datetime.now()
			d_t = (t_1 - t_0).seconds
			print('Calculation time for Layer {}: {} seconds.'.format(L,d_t))
	if True: #perform Kolmogorov-Smirnov analysis
		KS_Stat, KS_P = TwoSample2DKolmogorovSmirnov(R, T, Equalize = KS_Equalize, Plot=False)
		SumR, SumT = sum(R), sum(T)
		KS_Stats = [KS_Stat, KS_P, SumR, SumT]
	return LinRegressStats, MI_Stats, KS_Stats
        #LinRegressStats, MI_Stats, KS_Stats = GetMultiStats(Reference,Target, MI_NumLayers, MI_NonZero, KS_Equalize)

def FreqCompAngHist(x1,x2,dx1,dx2,NumTests):
    y1_Tests = []
    y2_Tests = []
    for i in range(NumTests):
        x = random.uniform(0,1)
        y1 = np.exp(-(x-x1)**2/dx1**2)
        y1_Tests.append(y1)
        y2 = np.exp(-(x-x2)**2/dx2**2)
        y2_Tests.append(y2)
    Title = 'Gaussian Test, x1={}, x2={}, dx1={}, dx2={}'.format(x1,x2,dx1,dx2)
    FreqPlot2(y1_Tests,y2_Tests,fit=True, avgline=True, staterr=False, title=Title, FontSize=16, CircleSize=10)
    AngleHist_Norm2(y1_Tests,y2_Tests, title=Title, avgline=True, FontSize=16, CircleSize=10)

def PltGaussianDistributions(x1,x2,dx1,dx2,NumTests):
	x = [i/(NumTests-1) for i in range(NumTests)]
	y1 = [np.exp(-(a-x1)**2/dx1**2) for a in x]
	y2 = [np.exp(-(a-x2)**2/dx2**2) for a in x]
	Title = 'Gaussian Distributions, x1={}, x2={}, dx1={}, dx2={}'.format(x1,x2,dx1,dx2)
	plt.plot(x,y1, c = 'Black')
	plt.plot(x,y2, '--', c = 'Red')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.title(Title)
	plt.show()

def SaveListOfListsData(SavePath, ListOfListsData, ListOfTitles):
	print('Saving list-of-lists data to file...')
	with open(SavePath, 'w', encoding='utf-8') as ff:
		print('\t'.join(ListOfTitles), file=ff) #Create header
		print('\t'.join(ListOfTitles))
		for i in range(len(ListOfListsData[0])):
			LineText = '\t'.join([str(x[i]).replace('\t',' ') for x in ListOfListsData])
			print(LineText, file=ff)
			if i <= 2:
				print(LineText)
			if i%10000 == 0:
				print('Number of lines saved: ', i)

def LoadListOfListsData(DataPath):
	print('Loading matrix data from file...')
	with open(DataPath, 'r', encoding='utf-8') as ff:
		ListLines = []
		DataLines = []
		header = ff.readline()
		Titles = header.strip().split('\t')
		NumElts = len(Titles)
		count = 0
		for line in ff:
			ListLines.append(line)
			count += 1
			LineElts = line.strip().split('\t')
			DataLines.append(LineElts)
			if count%10000 == 0:
			    print('Lines Loaded: {}'.format(count))
	Data = [[x[i] for x in DataLines] for i in range(len(DataLines[0]))]
	print(header.strip())
	print('\t'.join([str(x) for x in DataLines[0]]))
	print('\t'.join([str(x) for x in DataLines[1]]))
	return Data, Titles
	#Data = LoadListOfListsData(DataPath)

def LoadListOfListsData_Delim(DataPath, Delimiter):
	print('Loading matrix data from file...')
	with open(DataPath, 'r', encoding='utf-8') as ff:
		ListLines = []
		DataLines = []
		header = ff.readline()
		Titles = header.strip().split(Delimiter)
		NumElts = len(Titles)
		count = 0
		for line in ff:
			ListLines.append(line)
			count += 1
			LineElts = line.strip().split(Delimiter)
			DataLines.append(LineElts)
			if count%10000 == 0:
			    print('Lines Loaded: {}'.format(count))
	Data = [[x[i] for x in DataLines] for i in range(len(DataLines[0]))]
	print(header.strip())
	print('\t'.join([str(x) for x in DataLines[0]]))
	print('\t'.join([str(x) for x in DataLines[1]]))
	return Data, Titles
	#Data = LoadListOfListsData(DataPath)

def LoadListOfListsData_Floats(DataPath): #Original Version
	print('Loading matrix data from file...')
	with open(DataPath, 'r', encoding='utf-8') as ff:
		ListLines = []
		DataLines = []
		header = ff.readline()
		Titles = header.strip().split('\t')
		NumElts = len(Titles)
		count = 0
		for line in ff:
			ListLines.append(line)
			count += 1
			LineElts = line.strip().split('\t')
			DataLines.append(LineElts)
			if count%10000 == 0:
			    print('Lines Loaded: {}'.format(count))
	Data = [[float(x[i]) for x in DataLines] for i in range(len(DataLines[0]))]
	print(header.strip())
	print('\t'.join([str(x) for x in DataLines[0]]))
	print('\t'.join([str(x) for x in DataLines[1]]))
	return Data, Titles
	#Data, Titles = LoadListOfListsData_Floats(DataPath)

def LoadListOfListsData_Floats(DataPath): #This version handles presence of string entries
	print('Loading matrix data from file...')
	with open(DataPath, 'r', encoding='utf-8') as ff:
		ListLines = []
		DataLines = []
		header = ff.readline()
		Titles = header.strip().split('\t')
		NumElts = len(Titles)
		count = 0
		for line in ff:
			ListLines.append(line)
			count += 1
			LineElts = line.strip().split('\t')
			DataLines.append(LineElts)
			if count%10000 == 0:
			    print('Lines Loaded: {}'.format(count))
	Data = []
	for i in range(len(DataLines[0])):
		Data_List = []
		for x in DataLines:
			try:
				float_x = float(x[i])
				Data_List.append(float_x)
			except:
				Data_List.append(np.inf)
		Data.append(Data_List)
	#Data = [[float(x[i]) for x in DataLines] ]
	print(header.strip())
	print('\t'.join([str(x) for x in DataLines[0]]))
	#print('\t'.join([str(x) for x in DataLines[1]]))
	return Data
    
def GetFileInfo(FileName):
	FileTimeStamp = FileName[0:17]
	FileTitle = FileName[17:-4]
	FileExt = FileName[-4:]
	[KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent, ScaleAndType] = FileTitle.split(',')
	if MinDigitsTxt[0] == 'A':
		MinDigits = 0
	else:
		MinDigits = float(MinDigitsTxt[0])
	if 'MainBA' in BinningAndExtent:
		ExtentName = 'Main Buenos Aires'
		NumBins = float(BinningAndExtent.split('MainBA')[0].split('Bin')[1])
	elif 'CABA' in BinningAndExtent:
		ExtentName = 'Buenos Aires'
		NumBins = float(BinningAndExtent.split('CABA')[0].split('Bin')[1])
	if 'RelScale' in ScaleAndType:
		Absolute = False
	elif 'AbsScale' in ScaleAndType:
		Absolute = True
	if 'Squares' in ScaleAndType:
		Squares = True
	elif 'Circles' in ScaleAndType:
		Squares = False
	RelScale = 2
	return [KeyWord, UserType, TimeFrame, MinDigits, ExtentName, NumBins, Absolute, Squares, FileTimeStamp, FileTitle, FileExt]

if False: #Main Program
    if True: #Get Data
        drive = "E:\\"
        CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
        tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
    if False: #Bin Users
        ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
        [IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times, ID_names, ID_locations] = ID_User_Data
    if False: #Save Binned Data
        SavePath = 'E:\\Twitter\\Experiments\\Argentinian\\Results\\' + datetimestr() + '_User-Binned Tweets_Buenos Aires.txt'
        SaveBinnedData(SavePath, IDs, ID_counts, ID_lons, ID_lats, ID_texts, ID_dates, ID_times)
    if False: #Read Binned Data
        DataPath = 'E:\\Twitter\\Experiments\\Argentinian\\Results\\2021_1005_210032_User-Binned Tweets_Buenos Aires.txt'
        List_User_Elts, ListBadLines = LoadBinnedData(DataPath)
        ID_User_Data = GetUserData(List_User_Elts)
    if True: #Filter data by User Type
        UserType = 'All' #UserType options: 'All', 'NonStatic', 'Static', 'Mover'
        ID_Data = ID_User_Data
        ID_UserType_Inds = GetUserTypeInds(ID_Data,UserType)
        ID_UserType_Data = FilterByUserInds(ID_Data, ID_UserType_Inds)
        GeoData_All = CollectData(ID_UserType_Data)
    if False: #Filter users by location
        ID_UserType_Loc_Inds = GetRefWordUserInds(ID_UserType_Inds, ID_User_Data, LocationTokens)
        ID_UserType_Loc_Data = FilterByUserInds(ID_Data, ID_UserType_Loc_Inds)
    if True: #Filter data for ArgSp, PenSp or Bi-Dialectal users
        TokenListName = 'MaxList' #Options: 'MaxList', 'Unambiguous', 'GramLex', 'Formality', 'GenderNeutral'
        WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
        ID_Data = ID_UserType_Data
        ID_RefUser_Inds = GetTokenUserInds(ID_Data, RefTokens)
        ID_WordUser_Inds = GetTokenUserInds(ID_Data, WordTokens)
        ID_2DUser_Inds = list(set(ID_RefUser_Inds) & set(ID_WordUser_Inds))
        ID_RefUser_Data = FilterByUserInds(ID_Data, ID_RefUser_Inds)
        ID_WordUser_Data = FilterByUserInds(ID_Data, ID_WordUser_Inds)
        ID_2DUser_Data = FilterByUserInds(ID_Data, ID_2DUser_Inds)
        GeoData_Ref = CollectData(ID_RefUser_Data)
        GeoData_Word = CollectData(ID_WordUser_Data)
        GeoData_2D = CollectData(ID_2DUser_Data)
    if True: #Include only Tweets with selected timeframe and number of digits
        MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
        TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
        DigitTimeFilter = True
        GeoData_All_f = FilterDigitsAndTime(GeoData_All, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_Ref_f = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_Word_f = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_2D_f = FilterDigitsAndTime(GeoData_2D, MinDigits, TimeFrame, DigitTimeFilter)
    if True: #Include only Tweets with Ref & Word Tokens
        WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
        TokenFilter = True
        Include = True
        GeoData_Ref_f_ref = FilterByTokens(GeoData_Ref_f, RefTokens, TokenFilter, Include)
        GeoData_Word_f_word = FilterByTokens(GeoData_Word_f, WordTokens, TokenFilter, Include)
        GeoData_2D_ref_f = FilterByTokens(GeoData_2D_f, RefTokens, TokenFilter, Include)
        GeoData_2D_word_f = FilterByTokens(GeoData_2D_f, WordTokens, TokenFilter, Include)
    if True: #Include only Tweets with location in LocationTokens
        #OlgaLocationTokens = GetTokens('Location') #Locations provided by Olga
        #Locations extracted from the corpus: 'Local_Locations', 'Foreign_Locations', 'Argentina_Locations'
        LocationFilter = True
        Include = True
        GeoData_All_f_local = FilterByLocations(GeoData_All_f, Local_Locations, LocationFilter, Include)
        GeoData_All_f_foreign = FilterByLocations(GeoData_All_f, Foreign_Locations, LocationFilter, Include)
        GeoData_Ref_f_local = FilterByLocations(GeoData_Ref_f, Local_Locations, LocationFilter, Include)
        GeoData_Word_f_local = FilterByLocations(GeoData_Word_f, Local_Locations, LocationFilter, Include)
        GeoData_Ref_f_foreign = FilterByLocations(GeoData_Ref_f, Foreign_Locations, LocationFilter, Include)
        GeoData_Word_f_foreign = FilterByLocations(GeoData_Word_f, Foreign_Locations, LocationFilter, Include)
        GeoData_Ref_f_ref_local = FilterByLocations(GeoData_Ref_f_ref, Local_Locations, LocationFilter, Include)
        GeoData_Ref_f_ref_foreign = FilterByLocations(GeoData_Ref_f_ref, Foreign_Locations, LocationFilter, Include)
        GeoData_Word_f_word_local = FilterByLocations(GeoData_Word_f_word, Local_Locations, LocationFilter, Include)
        GeoData_Word_f_word_foreign = FilterByLocations(GeoData_Word_f_word, Foreign_Locations, LocationFilter, Include)        
    if True: #Include only Tweets with these tokens
        #KeepTokens = GetTokens('Locations')['Barrios_Olga']
        KeepTokens = GetTokens('Locations')['Barrios_Wikipedia_Tokens']
        BarriosFilter = True
        Include = True
        GeoData_Ref_f_Barrios = FilterByTokens(GeoData_Ref_f, KeepTokens, BarriosFilter, Include)
        GeoData_Word_f_Barrios = FilterByTokens(GeoData_Word_f, KeepTokens, BarriosFilter, Include)
    if True: #Eliminate Tweets with these tokens
        ElimTokens = ['"', "🎶"]
        Filter = True
        Include = False
        GeoData_Ref_f = FilterByTokens(GeoData_Ref_f, ElimTokens, Filter, Include)
        GeoData_Word_f = FilterByTokens(GeoData_Word_f, ElimTokens, Filter, Include)
##    if True: #Save Data
##        BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
##        Folder = now() + 'Extended Data\\'
##        SaveText(ID_Search_texts_ref_ext_f, BaseDir + Folder + now() + 'Extended Tweet Set_Ref_NonStatic_Pre06.2019_UnambigTokensLexGram.txt')
##        SaveText(ID_Search_texts_word_ext_f, BaseDir + Folder + now() + 'Extended Tweet Set_Word_NonStatic_Pre06.2019_UnambigTokensLexGram.txt')
    if True: #Prep for plotting
        cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
        osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
    if True: #MatrixBin and plot individual distributions
        GeoData_R = GeoData_Ref_f
        GeoData_W = GeoData_Word_f
        extent = GetExtent('Buenos Aires')
        NumBins = 100
        Scale = 0.2
        lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
        lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
        coord = [lons_ref_M, lats_ref_M]
        freq = np.array(counts_ref_M)
        mkscatplt2(coord, freq, extent, circ_scale = Scale, title = 'KMA', img='y', bdr='y', grd='n', clr='r', cmap='bwr', alpha = 0.3, osm_img = osm_img)
        freq = np.array(counts_word_M)
        mkscatplt2(coord, freq, extent, circ_scale = Scale, title = 'KMA', img='y', bdr='y', grd='n', clr='r', cmap='bwr', alpha = 0.3, osm_img = osm_img)
    if True: #MatrixBin and plot posneg
        ExtentName = 'Buenos Aires'
        extent = GetExtent(ExtentName)
        FilterTimeAndDigits = True
        NumBins = 100
        Absolute = False
        Squares = False
        RelScale = 2
        alpha = 0.4
        ElimZeros = True
        Show = True
        KeyWord = TokenListName+'_ArgSp vs PenSp'
        if FilterTimeAndDigits:
            GeoData_W = FilterByExtent(GeoData_Word_f, extent, Filter=True)
            GeoData_R = FilterByExtent(GeoData_Ref_f, extent, Filter=True)
            TFrame = TimeFrame
            MDigits = MinDigits
        else:
            GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
            GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
            TFrame = 'AllTimes'
            MDigits = 0
        ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_W, GeoData_R, ElimZeros)
        Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
        TitleElts = [KeyWord, UserType, TFrame, MDigits, NumBins, ExtentName, Absolute, Scale]
        fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
        FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1012_Max5Plots\\'
        if True:
            fig.savefig(fname = FigFolder+SaveTitle)
	#pltposneg_square_old(MarkerScale=77, NumBins, ID_pndata, extent, circ_scale=0, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
        #pltposneg(ID_pndata, extent, circ_scale=20000, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
    if True: #Get Statistics
        GeoData_R = GeoData_Ref_f
        GeoData_W = GeoData_Word_f
        extent = GetExtent('Buenos Aires')
        NumBins = 100
        title = 'KMA'
        lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
        lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
        FreqPlot(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title)
        AngleHist(counts_ref_M, counts_word_M, title = title, avgline=True)
        AngleHist_Norm(counts_ref_M, counts_word_M, title = title, avgline=True)

    ##### For Plotting MaxN #####
if False:
    if True: #MatrixBin and plot
        ExtentName = 'Buenos Aires'
        extent = GetExtent(ExtentName)
        GeoData_W = FilterByExtent(GeoData_Word_f, extent, Filter=True)
        GeoData_R = FilterByExtent(GeoData_Ref_f, extent, Filter=True)
        NumBins = 200
        Absolute = False
        Squares = False
        RelScale = 2
        vRef = True
        ElimZeros = True
        Show = True
        N = 5
        lons_word_M, lats_word_M, counts_word_M, Ew = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, [])
        lons_ref_M, lats_ref_M, counts_ref_M, Er = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, [])
        if Absolute:
            WmR = GetWmRabs(counts_word_M, counts_ref_M)
            #ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
        else:
            WmR = GetWmR(counts_word_M, counts_ref_M)
            #ID_pndata =  GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
        Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
        TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
        plottitle = GetTitle(TitleElts)
        ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_W, GeoData_R, ElimZeros)
        #pltposnegmax(WmR, lons_word_M, lats_word_M, N, extent, plottitle, ElimZeros)        
    if True:
        BlueCoord, RedCoord, BlueVals, RedVals, BlueIndices, RedIndices = GetRedBlueCoord(WmR, lons_word_M, lats_word_M)
        MaxNVals = RedVals[0:N]+BlueVals[0:N]
        MaxNLons = [RedCoord[i][0] for i in range(N)] + [BlueCoord[i][0] for i in range(N)]
        MaxNLats = [RedCoord[i][1] for i in range(N)] + [BlueCoord[i][1] for i in range(N)]
        MaxN_pndata = GetPosNegData(MaxNLons, MaxNLats, MaxNVals, ElimZeros)
        extent = GetExtent('Main Buenos Aires')
        fig = pltposneg(Show, MaxN_pndata, extent, circ_scale=Scale, title=plottitle, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.4, osm_img=osm_img)
        FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1012_Max5Plots\\'
        if False:
            fig.savefig(fname = FigFolder+SaveTitle[0:-4]+'_Max'+str(N)+'.png')
        fig, SaveTitle = PlotIt(Show, ID_pndata, TitleElts, Squares)
        if False:
            fig.savefig(fname = FigFolder+SaveTitle)
    if True:
        PlotLons, PlotLats, freq, extent_plot, extent_Tweets = PlotPoint(TweetCoords, PointCoord, LongBinSize, LatBinSize, NumBinsPlot, NumBinsTweets, Title, Hist, pltcolor, Alpha)

###### For Automating graphing process
if False: #MatrixBin and plot
    NumBinList = [50,75,100,200]
    AbsoluteList = [False, True]
    ExtentNameList = ['Buenos Aires', 'Main Buenos Aires']
    AbsRelScaleList = [0.1,0.2,0.4]
    RelRelScaleList = [1, 2, 4]
    FilterTimeAndDigits = True
    Squares1 = False
    Squares2 = True
    Squares = True
    RelScale = 2
    vRef = True
    ElimZeros = True
    Show = True
    Save = True
    UserType = 'Bilingual'
    KeyWord = 'Sp vs Eng'
    FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1018_Plots_Spanish v English\\'
    for ExtentName in ExtentNameList:
        extent = GetExtent(ExtentName)
        if FilterTimeAndDigits:
            GeoData_W = FilterByExtent(GeoData_2L_ES_f, extent, Filter=True)
            GeoData_R = FilterByExtent(GeoData_2L_EN_f, extent, Filter=True)
            TFrame = TimeFrame
            MDigits = MinDigits
        else:
            GeoData_W = FilterByExtent(GeoData_2L_ES, extent, Filter=True)
            GeoData_R = FilterByExtent(GeoData_2L_EN, extent, Filter=True)
            TFrame = 'AllTimes'
            MDigits = 0
        for NumBins in NumBinList:
            for Absolute in AbsoluteList:
                ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_W, GeoData_R, ElimZeros)
                if Absolute:
                    for RelScale in AbsRelScaleList:
                        Scale1 = GetScale(ExtentName, Squares1, Absolute, RelScale)
                        TitleElts1 = [KeyWord, UserType, TFrame, MDigits, NumBins, ExtentName, Absolute, Scale1]
                        fig1, SaveTitle1 = PlotIt(Show, ID_pndata, TitleElts1, Squares1)
                        if Save:
                            fig1.savefig(fname = FigFolder+SaveTitle1)
                    if Squares:
                        Scale2 = GetScale(ExtentName, Squares2, Absolute, RelScale)
                        TitleElts2 = [KeyWord, UserType, TFrame, MDigits, NumBins, ExtentName, Absolute, Scale2]
                        fig2, SaveTitle2 = PlotIt(Show, ID_pndata, TitleElts2, Squares2)
                        if Save:
                            fig2.savefig(fname = FigFolder+SaveTitle2)
                    plt.close('all')
                else:
                    for RelScale in RelRelScaleList:
                        Scale1 = GetScale(ExtentName, Squares1, Absolute, RelScale)
                        TitleElts1 = [KeyWord, UserType, TFrame, MDigits, NumBins, ExtentName, Absolute, Scale1]
                        fig1, SaveTitle1 = PlotIt(Show, ID_pndata, TitleElts1, Squares1)
                        if Save:
                            fig1.savefig(fname = FigFolder+SaveTitle1)
                    if Squares:
                        Scale2 = GetScale(ExtentName, Squares2, Absolute, RelScale)
                        TitleElts2 = [KeyWord, UserType, TFrame, MDigits, NumBins, ExtentName, Absolute, Scale2]
                        fig2, SaveTitle2 = PlotIt(Show, ID_pndata, TitleElts2, Squares2)
                        if Save:
                            fig2.savefig(fname = FigFolder+SaveTitle2)
                    plt.close('all')
    #pltposneg_square_old(MarkerScale=77, NumBins, ID_pndata, extent, circ_scale=0, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
    #pltposneg(ID_pndata, extent, circ_scale=20000, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)

###### For BiLingual Analysis
if False: #Filter users by user ID
    if True: #Load Spanish and English data
        drive = "E:\\"
        CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
        tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times = GetData(CorpusDir)
        ID_User_Data_ES, BinnedElts_ES = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times)
        CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\en\\Buenos Aires'
        tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times = GetData(CorpusDir)
        ID_User_Data_EN, BinnedElts_EN = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times)
    if True: # Get data from all users
        GeoData_ES = CollectData(ID_User_Data_ES)
        GeoData_EN = CollectData(ID_User_Data_EN)
    if True: # Get data from bilingual users
        BiLingualUserIDs = list(set(ID_User_Data_ES[0]) & set(ID_User_Data_EN[0]))
        ID_BiLingualUser_Inds_ES = GetUserInds(ID_User_Data_ES,BiLingualUserIDs)
        ID_BiLingualUser_Inds_EN = GetUserInds(ID_User_Data_EN,BiLingualUserIDs)
        ID_2LUser_Data_ES = FilterByUserInds(ID_User_Data_ES, ID_BiLingualUser_Inds_ES)
        ID_2LUser_Data_EN = FilterByUserInds(ID_User_Data_EN, ID_BiLingualUser_Inds_EN)
        GeoData_2L_ES = CollectData(ID_2LUser_Data_ES)
        GeoData_2L_EN = CollectData(ID_2LUser_Data_EN)
    if True: #Include only Tweets with selected timeframe and number of digits
        MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
        TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
        DigitTimeFilter = True
        GeoData_ES_f = FilterDigitsAndTime(GeoData_ES, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_EN_f = FilterDigitsAndTime(GeoData_EN, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_2L_ES_f = FilterDigitsAndTime(GeoData_2L_ES, MinDigits, TimeFrame, DigitTimeFilter)
        GeoData_2L_EN_f = FilterDigitsAndTime(GeoData_2L_EN, MinDigits, TimeFrame, DigitTimeFilter)
    	#pltposneg_square_old(MarkerScale=Scale, NumBins, ID_pndata, extent, circ_scale=0, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
        #pltposneg(ID_pndata, extent, circ_scale=Scale, title=title, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
    if True: #Include only Tweets with these tokens
        KeepTokensWord = ['Acaba de publicar']
        KeepTokensRef = ['Just posted']
        InstaFilter = True
        Include = True
        GeoData_ES_f_Insta = FilterByTokens(GeoData_ES_f, KeepTokensWord, InstaFilter, Include)
        GeoData_EN_f_Insta = FilterByTokens(GeoData_EN_f, KeepTokensRef, InstaFilter, Include)
    if True: #MatrixBin and plot
        ExtentName = 'Buenos Aires'
        extent = GetExtent(ExtentName)
        GeoData_W = FilterByExtent(GeoData_2L_ES, extent, Filter=True)
        GeoData_R = FilterByExtent(GeoData_2L_EN, extent, Filter=True)
        NumBins = 100
        Absolute = True
        Squares = True
        RelScale = 2
        vRef = True
        ElimZeros = True
        Show = True
        UserType = 'BiLingual'
        KeyWord = 'Engl vs Sp'
        ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_W, GeoData_R, ElimZeros)
        Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
        TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
        fig, SaveTitle = PlotIt(Show, ID_pndata, TitleElts, Squares)
        FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1014_Plots_English v Spanish\\'
        if False:
            fig.savefig(fname = FigFolder+SaveTitle)

###### For Gender Neutral Analysis
if False:
    if True: # Filter GeoData by Gender tokens
        GenderTokens = ["rxs ", "r@s ", "todx", "tod@", "unx ", "un@ ", "nxs", "n@s", "lxs", "l@s", "in@ ", "inx ", "much@", "muchx", "amigues", "amig@", "amigx", "l@ ", "lx ", "s/-a ", "s/a ", "r/-a ", "r/a ", "chiques", "amigues ", "lxs", "l@s"]
        GenderFilter = True
        Include = True
        GeoData_ES_Gender = FilterByTokens(GeoData_ES, GenderTokens, GenderFilter, Include)
    if True: # Collect a set of dates
        GenderDates = [getYMD_singlenum(x) for x in GeoData_ES_Gender[3]]
        GenderDatesYMD = [list(getYMDnum(x)) for x in GeoData_ES_Gender[3]]
        AllDates = [getYMD_singlenum(x) for x in GeoData_ES[3]]
        AllDatesYMD = [list(getYMDnum(x)) for x in GeoData_ES[3]]
    if True: # Bin by date
        GDates = []
        GCounts = []
        ADates = []
        ACounts = []
        BinStyle = 'by day'
        if BinStyle == 'by day':
            for date in GenderDates:
                if date not in GDates:
                    GDates.append(date)
                    GCounts.append(1)
                else:
                    i = GDates.index(date)
                    GCounts[i] += 1
            gender_dates = [x/10000 for x in GDates]
            for date in AllDates:
                if date not in ADates:
                    ADates.append(date)
                    ACounts.append(1)
                else:
                    i = ADates.index(date)
                    ACounts[i] += 1
            all_dates = [x/10000 for x in ADates]
        elif BinStyle == 'by month':
            for date in GenderDatesYMD:
                if date[0:2] not in GDates:
                    GDates.append(date[0:2])
                    GCounts.append(1)
                else:
                    i = GDates.index(date[0:2])
                    GCounts[i] += 1
            gender_dates = [x[0]+x[1]/100 for x in GDates]
            for date in AllDatesYMD:
                if date[0:2] not in ADates:
                    ADates.append(date[0:2])
                    ACounts.append(1)
                else:
                    i = ADates.index(date[0:2])
                    ACounts[i] += 1
            all_dates = [x[0]+x[1]/100 for x in ADates]
    if True: # sort dates
        GIndex = [i for i in range(len(gender_dates))]
        Sorted_GenderData = sorted(zip(gender_dates,GCounts))
        Sorted_GDates = [a for a,b in Sorted_GenderData]
        Sorted_GCounts = [b for a,b in Sorted_GenderData]
        AIndex = [i for i in range(len(all_dates))]
        Sorted_AllData = sorted(zip(all_dates,ACounts))
        Sorted_ADates = [a for a,b in Sorted_AllData]
        Sorted_ACounts = [b for a,b in Sorted_AllData]
        Sorted_RCounts = [Sorted_GCounts[i]/Sorted_ACounts[i] for i in range(len(Sorted_ACounts))]
    if True:
        plt.scatter(Sorted_Dates,Sorted_Counts, s = 3)
        plt.xlabel('Date in YYYY.MMDD format')
        plt.ylabel('Counts')
        plt.title('Counts of gender-neutral Tweets over time (by days)')
        plt.show()

##### For Context Map Plotting
def GetContexts():
	CONTEXT = {}
	CONTEXT['Data Folder'] = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1021_GoogleMaps Context Locations\\'
	CONTEXT['Museums'] = '2021_1021_GoogleMaps_CABA_Museums.txt'
	CONTEXT['Monuments'] = '2021_1021_GoogleMaps_CABA_Monuments.txt'
	CONTEXT['Attractions'] = '2021_1021_GoogleMaps_CABA_Attractions.txt'
	CONTEXT['Universities'] = '2021_1022_GoogleMaps_CABA_Universities.txt'
	CONTEXT['Schools'] = '2021_1022_GoogleMaps_CABA_Schools.txt'
	CONTEXT['English Name Schools'] = '2021_1022_GoogleMaps_CABA_English Name Schools.txt'
	CONTEXT['Soccer Stadiums'] = '2021_1022_GoogleMaps_CABA_Soccer Stadiums.txt'
	CONTEXT['Sports Clubs'] = '2021_1022_GoogleMaps_CABA_Sports Clubs.txt'
	CONTEXT['Govt Bldgs'] = '2021_1022_GoogleMaps_CABA_Govt Bldgs.txt'
	CONTEXT['Other Govt Bldgs'] = '2021_1022_GoogleMaps_CABA_Other Govt Bldgs.txt'
	CONTEXT['Comisarias'] = '2021_1022_GoogleMaps_CABA_Comisarias.txt'
	CONTEXT['CeSAC'] = '2021_1022_GoogleMaps_CABA_CeSAC.txt'
	CONTEXT['Starbucks'] = '2021_1023_GoogleMaps_CABA_Starbucks.txt'
	CONTEXT['Churches'] = '2021_1023_GoogleMaps_CABA_Churches.txt'
	CONTEXT['Hospitals'] = '2021_1028_GoogleMaps_CABA_Hospitals.txt'
	CONTEXT['Hospital Areas'] = '2021_1028_GoogleMaps_CABA_Hospital Area.txt'
	
	return CONTEXT

if False:
	if True: #get list of longitudes and latitudes from file
		ContextFileDir = CONTEXT['Data Folder']
		ContextName = 'Sports Clubs'
		ContextFileNames = [CONTEXT[ContextName]]
		title = ContextName + ' Context'
		locations, lon_str, lat_str = [],[],[]
		for file in ContextFileNames:
			[a,b,c] = LoadTabFile(ContextFileDir+file)
			locations = locations + a
			lon_str = lon_str + b
			lat_str = lat_str + c
		lons = [float(x) for x in lon_str]
		lats = [float(x) for x in lat_str]
	if True: #get bin indices associated with longitude and latitude list
		IndList =[]
		extent = GetExtent('Main Buenos Aires')
		NumBins = 200
		GeoData_W = GeoData_Word_f_word
		GeoData_R = GeoData_Ref_f_ref
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		for i in range(len(lons)):
			dist = [np.sqrt((lons[i] - lons_ref_M[j])**2 + (lats[i] - lats_ref_M[j])**2) for j in range(len(lons_ref_M))]
			IndList.append(dist.index(min(dist)))
	if True: #create pos-neg graph
		coord = [[lons_ref_M[ind] for ind in IndList], [lats_ref_M[ind] for ind in IndList]]
		counts_word_context = [counts_word_M[ind] for ind in IndList]
		counts_ref_context = [counts_ref_M[ind] for ind in IndList]
		Sum_word = sum(counts_word_M)
		Sum_ref = sum(counts_ref_M)
		WmR_context = [counts_word_context[i]/Sum_word - counts_ref_context[i]/Sum_ref for i in range(len(counts_word_context))]
		ID_pndata_context = GetPosNegData_Zeros(coord[0], coord[1], WmR_context)
		title1 = title+'_BinMainBA'+str(NumBins)+'_Circle-200'
		pltposneg_Zeros(Show, ID_pndata_context, extent, circ_scale=-200, title=title1, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', nullcolor='black', alpha=0.3, osm_img=osm_img)
		print('Red: {}/{} = {}%'.format(len(ID_pndata_context[0]), len(IndList), round(100*len(ID_pndata_context[0])/len(IndList))))
		print('Blue: {}/{} = {}%'.format(len(ID_pndata_context[1]), len(IndList), round(100*len(ID_pndata_context[1])/len(IndList))))
		print('Black: {}/{} = {}%'.format(len(ID_pndata_context[6]), len(IndList), round(100*len(ID_pndata_context[6])/len(IndList))))

######## For Locals & Foreigners Analysis #######
def GetLocations():
    LOCATION = {}
    LOCATION['Data Folder'] = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1024_Locals And Foreigners\\Locals vs Foreigners\\'
    LOCATION['LocalsAndForeigners'] = '2021_1024_List of Locations for Locals and Foreigners.txt'
    return LOCATION
    #LOCATION = GetLocations()

if False:
    if True: #get list of locations from file
        LOCATION = GetLocations()
        LocationFileDir = LOCATION['Data Folder']
        LocationName = 'LocalsAndForeigners'
        file = LOCATION[LocationName]
        title = LocationName + ' Locations'
        ColData,ListLines = LoadTSV_MixedLineFile(LocationFileDir+file)
        [num, locations, counts, local, foreign, argentina] = ColData
        Local_Locations, Local_Counts, Foreign_Locations, Foreign_Counts, Argentina_Locations, Argentina_Counts = [], [], [], [], [], []
        for i in range(len(locations)):
            if local[i] == '1':
                Local_Locations.append(locations[i])
                Local_Counts.append(int(counts[i]))
            if foreign[i] =='1':
                Foreign_Locations.append(locations[i])
                Foreign_Counts.append(int(counts[i]))
            if argentina[i] =='1':
                Argentina_Locations.append(locations[i])
                Argentina_Counts.append(int(counts[i]))
        print('{} local locations found with {} Tweets.'.format(len(Local_Locations), sum(Local_Counts)))
        print('{} foreign locations found with {} Tweets.'.format(len(Foreign_Locations), sum(Foreign_Counts)))
        print('{} Argentina locations found with {} Tweets.'.format(len(Argentina_Locations), sum(Argentina_Counts)))

###################Test Occupancy of Matrix###########################
def plt_square(Show, MarkerScale, NumBins, data, extent, circ_scale, title, img, bdr, grd, color, alpha, osm_img):
    lons = np.array(data[0])
    lats = np.array(data[1])
    freq = np.array(data[2])
    fig = plt.figure(figsize=(12,9)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    ax1.set_title(title,fontsize=16)
    if bdr == 'y':
        ax1.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax1.add_feature(cfeature.BORDERS, edgecolor="black")
    if grd == 'y':
        ax1.gridlines()
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(extent[0],extent[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(extent[2],extent[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_xlabel('Longitude', loc='center')
    ax1.set_ylabel('Latitude', loc='center')
    ax1.set_title(title)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    FigSize = plt.rcParams['figure.figsize']
    #NumLonBins, NumLatBins = GetNBins(extent, data)
    NumLonBins, NumLatBins = NumBins, NumBins
    A = AspectRatio(extent)*NumLonBins/NumLatBins
    #MarkerScale = 77 # for Buenos Aires
    #MarkerScale = 58 # for Main Buenos Aires
    if img == 'y':
        ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification
    if circ_scale < 0:
        circlesizes = [-circ_scale for x in freq]
    elif circ_scale == 0:
        circlesizes = [(MarkerScale*FigSize[0]/NumLonBins)**2 for x in freq]
    else:
        circlesizes = [abs(x)*circ_scale for x in freq]
    verts = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] #relative coordinates of marker
    RectVerts = [[x[0],x[1]*A] for x in verts]
    plt.scatter(x=lons, y=lats, color=color, s=circlesizes, alpha=alpha,transform=crs.PlateCarree(), marker=RectVerts, linewidth = 0)
    if Show:
        plt.show()
    return fig


######Plot Bins that have no counts or equal ref and word counts
if False:
	GeoData_R = GeoData_Ref_f_ref
	GeoData_W = GeoData_Word_f_word
	extent = GetExtent('Main Buenos Aires')
	NumBins = 75
	circ_scale = 0
	title = 'KMA'
	Show = True
	lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
	lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
	lons_equal = []
	lats_equal = []
	counts_equal = []
	for i in range(len(lons_ref_M)):
		if counts_ref_M[i] == counts_word_M[i]:
			lons_equal.append(lons_ref_M[i])
			lats_equal.append(lats_ref_M[i])
			counts_equal.append(counts_ref_M)
	lons_equal_non0 = []
	lats_equal_non0 = []
	counts_equal_non0 = []
	for i in range(len(lons_ref_M)):
		if counts_ref_M[i] == counts_word_M[i] and counts_word_M[i] != 0:
			lons_equal_non0.append(lons_ref_M[i])
			lats_equal_non0.append(lats_ref_M[i])
			counts_equal_non0.append(counts_ref_M)
	MarkerScale = 58
	data = [lons_equal_non0, lats_equal_non0, counts_equal_non0]
	fig = plt_square(Show, MarkerScale, NumBins, data, extent, circ_scale, title, img='y', bdr='y', grd='n', color='r', alpha=0.3, osm_img=osm_img)



##########Point Surrounding Analysis##########################
def PointsOfInterest():
    POINT = {}
    POINT['Estadio Antonio V. Liberti'] = [-58.44977, -34.5453]
    POINT['Estadio La Bombonera'] = [-58.36475, -34.63561]
    POINT['Estadio Libertadores de America'] = [-58.37134, -34.67021]
    POINT['Estadio José Amalfitani'] = [-58.52069, -34.6354]
    POINT['Hospital Neuropsiquiátrico Braulio A. Moyano'] = [-58.38384, -34.64049]
    POINT['Plaza de Mayo'] = [-58.37228, -34.60836]
    POINT['Obelisco'] = [-58.38157, -34.60373]
    POINT['Plaza del Congreso'] = [-58.389760, -34.609695]
    return POINT

def BinByCoord(GeoData):
    PlotCoords = [[GeoData[0][i], GeoData[1][i]] for i in range(len(GeoData[0]))]
    HistPlotCoords, freq = GetHist(PlotCoords)
    PlotLons = [x[0] for x in HistPlotCoords]
    PlotLats = [x[1] for x in HistPlotCoords]
    return PlotLons, PlotLats, freq

if False: #MatrixBin and plot
	ExtentName = 'Main Buenos Aires'
	extent = GetExtent(ExtentName)
	FilterTimeAndDigits = True
	NumBins = 200
	Absolute = False
	Squares = False
	RelScale = 2
	ElimZeros = True
	Show = True
	BinType = 'Coordinates' #Options: 'Coordinates', 'Matrix'
	GeoData_W = FilterByExtent(GeoData_Word_f_word, extent, Filter=True)
	GeoData_R = FilterByExtent(GeoData_Ref_f_ref, extent, Filter=True)
	if BinType == 'Coordinates':
		ID_pndata = BinByCoord(GeoData_W) + BinByCoord(GeoData_R)
		RelScale = RelScale/1000
		KeyWord = TokenListName+'_ArgSp & PenSp'
	elif BinType == 'Matrix':
		ID_pndata = FilterMatrixBinPlot(extent, NumBins, Absolute, GeoData_W, GeoData_R, ElimZeros)
		KeyWord = TokenListName+'_ArgSp vs PenSp'
	Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
	TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale, BinType]
	title = GetTitle2(TitleElts)
	PointName = 'Obelisco'
	PointCoord = POINT[PointName]
	Range = 0.005 #in degrees
	extent1 = [PointCoord[0]-Range, PointCoord[0]+Range, PointCoord[1]-Range, PointCoord[1]+Range]
	#extent1 = GetExtent('Main Buenos Aires')
	if extent1 != extent:
		zoom = '_Zoom_' + PointName + str(Range)
	else:
		zoom = ''
	if Squares:
		title1 = title + 'Squares' + zoom
		MarkerScale = Scale
		circ_scale = 0
		fig = pltposneg_square_old(Show, MarkerScale, NumBins, ID_pndata, extent1, circ_scale, title1, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
	else:
		title1 = title + 'Circles' + zoom
		circ_scale = Scale
		fig = pltposneg(Show, ID_pndata, extent1, circ_scale, title1, img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=0.3, osm_img=osm_img)
	#fig, SaveTitle = PlotIt(Show, ID_pndata, TitleElts, Squares)
	FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1029_Localized Plotting\\'
	SaveTitle = datetimestr()+'_'+title1+'.png'
	if True:
		fig.savefig(fname = FigFolder+SaveTitle)

############ Plot Distributions ###################
if False: #Get the distributions of all, ArgSp and PenSp
    if True: #MatrixBin and plot individual distributions
        GeoData_R = GeoData_All_f_ES
        extent = GetExtent('Buenos Aires')
        TimeStamp = datetimestr()+'_'
        NumBins = 100
        #GeoData_W = GeoData_Word_f
        lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
        #lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
        coord = [lons_ref_M, lats_ref_M]
        freq = np.array(counts_ref_M)
        title = 'Spanish Tweets, AllTimes, 5+Digits, Bin{}CABA+, Circles0.01'.format(NumBins)
        fig = mkscatplt2(coord, freq, extent, circ_scale = 0.01, title = title, img='y', bdr='y', grd='n', clr='purple', cmap='bwr', alpha = 0.4, osm_img = osm_img)
        fig.savefig(fname = FigFolder+TimeStamp+title+'.png', dpi=300)
    if True: #MatrixBin and plot individual distributions
        GeoData_R = GeoData_Ref_f_ref_PenSp
        GeoData_W = GeoData_Word_f_word_ArgSp
        extent = GetExtent('Buenos Aires')
        NumBins = 100
        Scale = 1
        lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
        lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
        TimeStamp = datetimestr()+'_'
        KeyWord = TokenListName+' RefSp Tweets'
        TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
        title = GetTitleBase(TitleElts)
        coord = [lons_ref_M, lats_ref_M]
        freq = np.array(counts_ref_M)
        fig = mkscatplt2(coord, freq, extent, circ_scale = Scale, title = title + ',Bin{}CABA+, Circles{}'.format(NumBins,Scale), img='y', bdr='y', grd='n', clr='b', cmap='bwr', alpha = 0.4, osm_img = osm_img)
        fig.savefig(fname = FigFolder+TimeStamp+title+'.png', dpi=300)
        KeyWord = TokenListName+' ArgSp Tweets'
        TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
        title = GetTitleBase(TitleElts)
        freq = np.array(counts_word_M)
        fig = mkscatplt2(coord, freq, extent, circ_scale = Scale, title = 'ArgSp Tweets, AllTimes, 5+Digits, Bin{}CABA+, Circles{}'.format(NumBins,Scale), img='y', bdr='y', grd='n', clr='r', cmap='bwr', alpha = 0.4, osm_img = osm_img)
        fig.savefig(fname = FigFolder+TimeStamp+title+'.png', dpi=300)


############ Get Differential Distributions for Spanish vs English ###############
if False:
	if True: #Get English and Spanish Tweets
		GeoData_Ref = GeoData_All_EN
		GeoData_Word = GeoData_All_ES
	if True:
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData_Ref_f = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
		GeoData_Word_f = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, Filter = True)
		GeoData_Ref_f_ref = FilterByTokens(GeoData_Ref_f, RefTokens, Filter = False, Include = True)
		GeoData_Word_f_word = FilterByTokens(GeoData_Word_f, WordTokens, Filter = False, Include = True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 100
		Absolute = False
		Squares = True
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		KeyWord = 'Spanish vs English Users'
		GeoData_W = FilterByExtent(GeoData_Word_f_word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref_f_ref, extent, Filter=True)
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)


############ Get Differential Distribution, FreqPlot and Angle Histograms
if False:
	if True: #Filter data for ArgSp, PenSp or Bi-Dialectal users
		TokenListName = 'Null_vb' # Options: 'Null_vb', 'Null_loslas', 'BocaPalermo', 'Stadiums', 'TangoFutbol', 'Formality', 'MaxList'
		WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
		ID_Data = ID_UserType_Data_ES
		ID_RefUser_Inds = GetTokenUserInds(ID_Data, RefTokens)
		ID_WordUser_Inds = GetTokenUserInds(ID_Data, WordTokens)
		ID_RefUser_Data = FilterByUserInds(ID_Data, ID_RefUser_Inds)
		ID_WordUser_Data = FilterByUserInds(ID_Data, ID_WordUser_Inds)
		GeoData_Ref = CollectData(ID_RefUser_Data)
		GeoData_Word = CollectData(ID_WordUser_Data)
	if True:
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData_Ref_f = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
		GeoData_Word_f = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, Filter = True)
		GeoData_Ref_f_ref = FilterByTokens(GeoData_Ref_f, RefTokens, Filter = True, Include = True)
		GeoData_Word_f_word = FilterByTokens(GeoData_Word_f, WordTokens, Filter = True, Include = True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Main Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 75
		Absolute = False
		Squares = False
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		KeyWord = TokenListName+' Tweets'
		GeoData_W = FilterByExtent(GeoData_Word_f_word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref_f_ref, extent, Filter=True)
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)
		
################### Local vs Foreign Analysis ##########################
if False:
	if True: #Filter users by location
		ID_UserType_ForeignUser_Inds = GetLocationUserTypeInds(ID_UserType_Inds_ES, ID_User_Data_ES, Foreign_Locations)
		ID_UserType_LocalUser_Inds = GetLocationUserTypeInds(ID_UserType_Inds_ES, ID_User_Data_ES, Local_Locations)
		ID_UserType_ForeignUser_Data = FilterByUserInds(ID_User_Data_ES, ID_UserType_ForeignUser_Inds)
		ID_UserType_LocalUser_Data = FilterByUserInds(ID_User_Data_ES, ID_UserType_LocalUser_Inds)
		GeoData_Ref = CollectData(ID_UserType_ForeignUser_Data)
		GeoData_Word = CollectData(ID_UserType_LocalUser_Data)
	if True: #Filter Digits and TimeFrame
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData_Ref_f = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
		GeoData_Word_f = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, Filter = True)
		GeoData_Ref_f_ref = FilterByTokens(GeoData_Ref_f, RefTokens, Filter = False, Include = True)
		GeoData_Word_f_word = FilterByTokens(GeoData_Word_f, WordTokens, Filter = False, Include = True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 100
		Absolute = False
		Squares = True
		RelScale = 2
		alpha = 0.4
		ElimZeros, Show = True, True
		KeyWord = 'Local vs Foreign Tweets' #TokenListName+' Tweets'
		GeoData_W = FilterByExtent(GeoData_Word_f_word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref_f_ref, extent, Filter=True)
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
			SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
			SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
	if True: #Get Statistics
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)


########## Load data from file and create figure plots ############################
if False: #Load file and create figures
        if True: #Get list of files
                LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
                FigFiles = os.listdir(LoadFolder)
                DatFiles =  [f for f in FigFiles if '.dat' in f]
        if True: #Prep for plotting
                cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
                osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
        if True: #Select Specific File
                DataFileIndex = 0
                FileName = DatFiles[DataFileIndex]
                Data = LoadMatrixData(LoadFolder+FileName)
                [lons_word_M, lats_word_M, counts_ref_M, counts_word_M] = Data
                FileTimeStamp = FileName[0:17]
                FileTitle = FileName[17:-4]
                FileExt = FileName[-4:]
                [KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent, ScaleAndType] = FileTitle.split(',')
                if MinDigitsTxt[0] == 'A':
                    MinDigits = 0
                else:
                    MinDigits = float(MinDigitsTxt[0])
                if 'MainBA' in BinningAndExtent:
                    ExtentName = 'Main Buenos Aires'
                    NumBins = float(BinningAndExtent.split('MainBA')[0].split('Bin')[1])
                elif 'CABA' in BinningAndExtent:
                    ExtentName = 'Buenos Aires'
                    NumBins = float(BinningAndExtent.split('CABA')[0].split('Bin')[1])
                if 'RelScale' in ScaleAndType:
                    Absolute = False
                    RelScale = 2
                if 'Squares' in ScaleAndType:
                    Squares = True
        if True: #MatrixBin and plot posneg
                extent = GetExtent(ExtentName)
                alpha = 0.4
                ElimZeros = True
                Show = True
                KeyWord = 'Spanish vs English Users'
                if Absolute:
                        WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
                        ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
                else:
                        WmR = GetWmR(counts_word_M, counts_ref_M)
                        ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
                Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
                TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
                fig_M, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
        if True: #Get Statistics Figures
                title = GetTitleBase(TitleElts)
                fig_FP,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
                fig_NAH,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
        if True: #Save Figures
                AnalysisTimeStamp = datetimestr()+'_'
                SaveFolder = smartdir(LoadFolder + AnalysisTimeStamp + 'Analysis') + '\\'
                fig_M.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle, dpi=300)
                fig_FP.savefig(fname = SaveFolder+FileTimeStamp+title+'_FreqPlot.png', dpi=300)
                fig_NAH.savefig(fname = SaveFolder+FileTimeStamp+title+'_NormAngHist.png', dpi=300)

##################Load Data from file, create figure plots and calculate Moran's I vs Neighborhood radius#####################
if False: #Load file and create figures
	if True: #Get list of files
		#LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
		LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Moran\'s I Calculation\\'
		FigFiles = os.listdir(LoadFolder)
		DatFiles =  [f for f in FigFiles if '.dat' in f]
		print('Number of data files: {}'.format(len(DatFiles)))
	if False: #Prep for plotting
		cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
		osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
	if True: #Select Specific File
		DataFileIndex = 0
		FileName = DatFiles[DataFileIndex]
		print('File name: {}'.format(FileName))
		Data = LoadMatrixData(LoadFolder+FileName)
		[lons_word_M, lats_word_M, counts_ref_M, counts_word_M] = Data
		FileTimeStamp = FileName[0:17]
		FileTitle = FileName[17:-4]
		FileExt = FileName[-4:]
		[KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent, ScaleAndType] = FileTitle.split(',')
		if MinDigitsTxt[0] == 'A':
		    MinDigits = 0
		else:
		    MinDigits = float(MinDigitsTxt[0])
		if 'MainBA' in BinningAndExtent:
		    ExtentName = 'Main Buenos Aires'
		    NumBins = float(BinningAndExtent.split('MainBA')[0].split('Bin')[1])
		elif 'CABA' in BinningAndExtent:
		    ExtentName = 'Buenos Aires'
		    NumBins = float(BinningAndExtent.split('CABA')[0].split('Bin')[1])
		if 'RelScale' in ScaleAndType:
		    Absolute = False
		    RelScale = 2
		if 'Squares' in ScaleAndType:
		    Squares = True
	if True: #MatrixBin and plot posneg
		extent = GetExtent(ExtentName)
		alpha = 0.4
		ElimZeros = True
		Show = True
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig_M, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
	if True: #Get Statistics Figures
		title = GetTitleBase(TitleElts)
		fig_FP,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig_NAH,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
	if False: #Save Figures
		AnalysisTimeStamp = datetimestr()+'_'
		SaveFolder = smartdir(LoadFolder + AnalysisTimeStamp + 'Analysis') + '\\'
		fig_M.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle, dpi=300)
		fig_FP.savefig(fname = SaveFolder+FileTimeStamp+title+'_FreqPlot.png', dpi=300)
		fig_NAH.savefig(fname = SaveFolder+FileTimeStamp+title+'_NormAngHist.png', dpi=300)
	if True: #Calculate Moran's I statistic
		R, T = Data[2], Data[3]
		RBar, TBar = np.mean(R), np.mean(T)
		N = len(R)
		D_values = [T[i]/TBar - R[i]/RBar for i in range(N)]
		Layers = 20
		MI_Stats = []
		for L in range(1,Layers+1,1):
			t_0 = datetime.now()
			print('Layer number {}'.format(L))
			MI = MoranI(D_values, L, "Circle")
			MI_Stats.append(MI)
			print(MI)
			t_1 = datetime.now()
			d_t = (t_1 - t_0).seconds
			print('Calculation time for Layer {}: {} seconds.'.format(L,d_t))
	if True:
		x_val = [i for i in range(1,Layers+1,1)]
		plt.plot(x_val,[x[0] for x in MI_Stats], c = 'black')
		plt.xlabel('Neighborhood Radius [pix]')
		plt.ylabel('Moran\'s I')
		plt.title('Value of Moran\'s I vs. Neighborhood Size')
		plt.show()

################## Get Summary of Moran's I statistics ######################
if False: #Load file and create figures
        if True: #Get list of files
                LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Moran\'s I Calculation\\MoranIStats_NonZero\\'
                FigFiles = os.listdir(LoadFolder)
                DatFiles =  [f for f in FigFiles if '.dat' in f]
                print('Number of data files: {}'.format(len(DatFiles)))
                for i in range(len(FigFiles)):
                        print('File index: {}, FileName: {}'.format(i,DatFiles[i]))
        if True: #Loop over desired files
                FileIndices = [4,5,6,14,7,0,11,3,2]
                NumLayers_List, MoranI_List = [], []
                for i in FileIndices:
                        DataFileIndex = i
                        FileName = DatFiles[DataFileIndex]
                        print('File name: {}'.format(FileName))
                        [NumLayers, MoranI] = LoadMoranIData(LoadFolder+FileName)
                        NumLayers_List.append(NumLayers)
                        MoranI_List.append(MoranI)
        if True:
            for i in range(len(NumLayers_List)):
                    plt.plot(NumLayers_List[i], MoranI_List[i])
            plt.xlabel('Number of Layers')
            plt.ylabel('Moran\'s I')
            plt.title('Summary of Moran\'s I statistics vs. Neighborhood Size')
            plt.show()
        if True: #Plot Moran's I vs. Regression Value
            RegressionR_List = [0.0009, 0.001, 0.220, 0.302, 0.515, 0.936, 0.977, 0.984, 0.991]
            plt.plot(RegressionR_List, [MI[0] for MI in MoranI_List])
            plt.xlabel('Regression r-value')
            plt.ylabel('Moran\'s I for 1 Layer')
            plt.title('Connection between Moran\'s I and Regression value')
            plt.show()

#############Kolmogorov-Smirnov Test#######################################
if False: #Load file and calculate 2D Kolmogorov-Smirnov
        if True: #Get list of files
                #LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
                LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Moran\'s I Calculation\\'
                FigFiles = os.listdir(LoadFolder)
                DatFiles =  [f for f in FigFiles if '.dat' in f]
                print('Number of data files: {}'.format(len(DatFiles)))
        if True: #Select Specific File
                DataFileIndex = 2
                FileName = DatFiles[DataFileIndex]
                print('File name: {}'.format(FileName))
                Data = LoadMatrixData(LoadFolder+FileName)
                TwoSample2DKolmogorovSmirnov(Data[2], Data[3])
        if True: #Compare results of 2D Kolmogorov-Smirnov test to Regression values
                MaxDiffs = [0.8953285052521692, 0.9994932860400304, 0.2587710385334041, 0.13416562955877231, 0.21085271317829624, 0.10347423057915514, 0.02971653826160403, 0.024454065150853133, 0.009682282161053823]
                RegressionR_List = [0.0009, 0.001, 0.220, 0.302, 0.515, 0.936, 0.977, 0.984, 0.991]
                plt.plot(RegressionR_List, MaxDiffs)
                plt.xlabel('Regression r-value')
                plt.ylabel('Max Difference of Cummulative Prob. Distrib.')
                plt.title('Frequency Comparison vs. Kolmogorov-Smirnov Test')
                plt.show()
        if True: #Plot Moran's I vs. Regression Value
                MIs = [MI[0] for MI in MoranI_List]
                plt.scatter(MIs, MaxDiffs, s = 20)
                if True: #Add best fit line
                        m,b = np.polyfit(MIs,MaxDiffs,1)
                        plt.plot(MIs, [m*X+b for X in MIs], color = 'red')
                plt.xlabel('Moran\'s I for 1 Layer')
                plt.ylabel('Max Difference of Cummulative Prob. Distrib.')
                plt.title('Connection between KS-Test and Moran\'s I')
                plt.show()

if False: #Load file and calculate 2D Kolmogorov-Smirnov
        if True: #Get list of files
                #LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\2021_1124_Data for Article 1\\'
                LoadFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Data\\'
                FigFiles = os.listdir(LoadFolder)
                DatFiles =  [f for f in FigFiles if '.dat' in f]
                print('Number of data files: {}'.format(len(DatFiles)))
        if True: #Loop over desired files
                FileIndices = [4,5,6,14,7,0,11,3,2]
                MaxDiffs = [0.8953285052521692, 0.9994932860400304, 0.2587710385334041, 0.13416562955877231, 0.21085271317829624, 0.10347423057915514, 0.02971653826160403, 0.024454065150853133, 0.009682282161053823]
                RegressionR_List = [0.0009, 0.001, 0.220, 0.302, 0.515, 0.936, 0.977, 0.984, 0.991]
                KS_Stats,KS_Ps,SumRs,SumTs = [], [], [], []
                for i in FileIndices:
                        FileName = DatFiles[i]
                        print('File name: {}'.format(FileName))
                        Data = LoadMatrixData(LoadFolder+FileName)
                        KS_Stat, KS_P = TwoSample2DKolmogorovSmirnov(Data[2], Data[3], Equalize=False,Plot=False)
                        KS_Stats.append(KS_Stat)
                        KS_Ps.append(KS_P)
                        SumR, SumT = sum(Data[2]), sum(Data[3])
                        SumRs.append(SumR)
                        SumTs.append(SumT)
        if True:
                SumRs = [15966, 4131, 1389, 62143, 516, 15536, 185437, 30463, 383545]
                SumTs = [585, 3978, 1715, 649501, 2621, 453504, 320838, 38136, 318240]
                KS_Ps = [0.0, 0.0, 5.979989456562977e-39, 0.0, 8.66079453908716e-06, 1.2071128637387677e-194, 3.647896454423924e-98, 7.1861458325765964e-06, 3.0422163637662123e-15]
                KS_Stats = [0.887746075795681, 0.9990317114500117, 0.24040199233040815, 0.1410731116068762, 0.11967812155251695, 0.12200673131816334, 0.03094417503655833, 0.01923880931367028, 0.009903798382011275]

####################Filter by Tokens######################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
	if False:
		GeoData = CollectData(ID_User_Data)
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if False: #Include only Tweets with Ref & Word Tokens
		TokenListName = 'Formality2' # Options: 'Null_vb', 'Null_loslas', 'BocaPalermo', 'Stadiums', 'TangoFutbol', 'Formality', 'MaxList'
		WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
		GeoData_Ref = FilterByTokens(GeoData, RefTokens, Filter=True, Include=True)
		GeoData_Word = FilterByTokens(GeoData, WordTokens, Filter=True, Include=True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 100
		Absolute = False
		Squares = True
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		KeyWord = TokenListName+' Tweets'
		GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
		if True: #Randomly filter the larger data set to equalize the numbers
			N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
			if N_W > N_R:
				RandInds = GetRandomInds(N_W,N_R)
				GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
				KeyWord = 'Equalized ' + KeyWord
			if N_R > N_W:
				RandInds = GetRandomInds(N_R,N_W)
				GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				KeyWord = 'Equalized ' + KeyWord
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0409_Data for Rebuttal\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)

##########################ArgSp vs PenSp Users##############################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
	if True: #Filter data for ArgSp, PenSp or Bi-Dialectal users
		TokenListName = 'MaxList' #Options: 'MaxList', 'Unambiguous', 'GramLex', 'Formality', 'GenderNeutral'
		WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
		ID_RefUser_Inds = GetTokenUserInds(ID_User_Data, RefTokens)
		ID_WordUser_Inds = GetTokenUserInds(ID_User_Data, WordTokens)
		ID_RefUser_Data = FilterByUserInds(ID_User_Data, ID_RefUser_Inds)
		ID_WordUser_Data = FilterByUserInds(ID_User_Data, ID_WordUser_Inds)
		GeoData_Ref = CollectData(ID_RefUser_Data)
		GeoData_Word = CollectData(ID_WordUser_Data)
	if True:
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData_Ref = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
		GeoData_Word = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, Filter = True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Main Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 75
		Absolute = False
		Squares = False
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		KeyWord = TokenListName+' Tweets'
		GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
		if False: #Randomly filter the larger data set to equalize the numbers
			N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
			if N_W > N_R:
				RandInds = GetRandomInds(N_W,N_R)
				GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
				KeyWord = 'Equalized ' + KeyWord
			if N_R > N_W:
				RandInds = GetRandomInds(N_R,N_W)
				GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				KeyWord = 'Equalized ' + KeyWord
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0409_Data for Rebuttal\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)

##############################Local vs Foreign Analysis#################################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
	if False:
		GeoData = CollectData(ID_User_Data)
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if True: #get list of locations from file
		LOCATION = GetLocations()
		LocationFileDir = LOCATION['Data Folder']
		LocationName = 'LocalsAndForeigners'
		file = LOCATION[LocationName]
		title = LocationName + ' Locations'
		ColData,ListLines = LoadTSV_MixedLineFile(LocationFileDir+file)
		[num, locations, counts, local, foreign, argentina] = ColData
		Local_Locations, Local_Counts, Foreign_Locations, Foreign_Counts, Argentina_Locations, Argentina_Counts = [], [], [], [], [], []
		for i in range(len(locations)):
			if local[i] == '1':
				Local_Locations.append(locations[i])
				Local_Counts.append(int(counts[i]))
			if foreign[i] =='1':
				Foreign_Locations.append(locations[i])
				Foreign_Counts.append(int(counts[i]))
			if argentina[i] =='1':
				Argentina_Locations.append(locations[i])
				Argentina_Counts.append(int(counts[i]))
		print('{} local locations found with {} Tweets.'.format(len(Local_Locations), sum(Local_Counts)))
		print('{} foreign locations found with {} Tweets.'.format(len(Foreign_Locations), sum(Foreign_Counts)))
		print('{} Argentina locations found with {} Tweets.'.format(len(Argentina_Locations), sum(Argentina_Counts)))
	if True: #Include only Tweets with location in LocationTokens
		#Locations extracted from the corpus: 'Local_Locations', 'Foreign_Locations', 'Argentina_Locations'
		GeoData_Ref = FilterByLocations(GeoData, Foreign_Locations, Filter=True, Include=True)
		GeoData_Word = FilterByLocations(GeoData, Local_Locations, Filter=True, Include=True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Main Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 75
		Absolute = False
		Squares = False
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		#KeyWord = TokenListName+' Tweets'
		KeyWord = 'Local vs Foreign Tweets'
		GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
		if False: #Randomly filter the larger data set to equalize the numbers
			N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
			if N_W > N_R:
				RandInds = GetRandomInds(N_W,N_R)
				GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
				KeyWord = 'Equalized ' + KeyWord
			if N_R > N_W:
				RandInds = GetRandomInds(N_R,N_W)
				GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				KeyWord = 'Equalized ' + KeyWord
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0409_Data for Rebuttal\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)


#################################Spanish vs English Tweets########################################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
	if False:
		GeoData = CollectData(ID_User_Data)
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if False: #Get English and Spanish Tweets
		CorpusDir_EN = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\en\\Buenos Aires'
		tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN = GetData(CorpusDir_EN)
		ID_User_Data_EN, BinnedElts_EN = QuickBin(tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN)
		GeoData_Ref = CollectData(ID_User_Data_EN)
		GeoData_Ref = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
		GeoData_Word = GeoData
	if False: #Save Corpus Data
		ListOfTitles = 'tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations'.split(',')
		#CorpusData = [tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations]
		#CorpusSavePath = "C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Data\\Corpus Data\\ExtentBasedGeoLocationCorpora_es_Buenos Aires.txt"
		#SaveListOfListsData(CorpusSavePath, CorpusData, ListOfTitles)
		CorpusData_EN = [tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN]
		CorpusSavePath_EN = "C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Data\\Corpus Data\\ExtentBasedGeoLocationCorpora_en_Buenos Aires.txt"
		SaveListOfListsData(CorpusSavePath_EN, CorpusData_EN, ListOfTitles)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Main Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 75
		Absolute = False
		Squares = False
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		#KeyWord = TokenListName+' Tweets'
		KeyWord = 'Spanish vs English Tweets'
		GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
		if False: #Randomly filter the larger data set to equalize the numbers
			N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
			if N_W > N_R:
				RandInds = GetRandomInds(N_W,N_R)
				GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
				KeyWord = 'Equalized ' + KeyWord
			if N_R > N_W:
				RandInds = GetRandomInds(N_R,N_W)
				GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				KeyWord = 'Equalized ' + KeyWord
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0409_Data for Rebuttal\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if True: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)

#######################################Spanish vs English Users####################################################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		CorpusDir_EN = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\en\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
		tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN = GetData(CorpusDir_EN)
	if False: #Save Corpus Data
		ListOfTitles = 'tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations'.split(',')
		CorpusData = [tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations]
		CorpusSavePath = "C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Data\\Corpus Data\\ExtentBasedGeoLocationCorpora_es_Buenos Aires.txt"
		SaveListOfListsData(CorpusSavePath, CorpusData, ListOfTitles)
		CorpusData_EN = [tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN]
		CorpusSavePath_EN = "C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\Data\\Corpus Data\\ExtentBasedGeoLocationCorpora_en_Buenos Aires.txt"
		SaveListOfListsData(CorpusSavePath_EN, CorpusData_EN, ListOfTitles)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
		ID_User_Data_EN, BinnedElts_EN = QuickBin(tweet_IDs_EN, tweet_lons_EN, tweet_lats_EN, tweet_texts_EN, tweet_dates_EN, tweet_times_EN, tweet_names_EN, tweet_locations_EN)
		CData = [CorpusData[i] + CorpusData_EN[i] for i in range(len(CorpusData))]
		ID_User_Data_C, BinnedElts_C = QuickBin(CData[0], CData[1], CData[2], CData[3], CData[4], CData[5], CData[6], CData[7])
	if False: #Get Spanish and English Users
		#ID_User_Data_Combined = [ID_User_Data[i] + ID_User_Data_EN[i] for i in range(len(ID_User_Data))]
		IDs_ES = list(set(tweet_IDs))
		IDs_EN = list(set(tweet_IDs_EN))
		ID_User_Inds_ES = GetUserIDInds(ID_User_Data_C, IDs_ES)
		ID_User_Inds_EN = GetUserIDInds(ID_User_Data_C, IDs_EN)
		ID_User_Data_ES = FilterByUserInds(ID_User_Data_C, ID_User_Inds_ES)
		ID_User_Data_EN = FilterByUserInds(ID_User_Data_C, ID_User_Inds_EN)
		GeoData_Word = CollectData(ID_User_Data_ES)
		GeoData_Ref = CollectData(ID_User_Data_EN)
	if False: #Filter by digits and by time-frame
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData_Word = FilterDigitsAndTime(GeoData_Word, MinDigits, TimeFrame, Filter = True)
		GeoData_Ref = FilterDigitsAndTime(GeoData_Ref, MinDigits, TimeFrame, Filter = True)
	if True: #MatrixBin and plot posneg
		ExtentName = 'Main Buenos Aires'
		extent = GetExtent(ExtentName)
		FilterTimeAndDigits = True
		NumBins = 75
		Absolute = False
		Squares = True
		RelScale = 2
		alpha = 0.4
		ElimZeros = True
		Show = True
		#KeyWord = TokenListName+' Tweets'
		KeyWord = 'Spanish vs English Users'
		GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
		GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
		if False: #Randomly filter the larger data set to equalize the numbers
			N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
			if N_W > N_R:
				RandInds = GetRandomInds(N_W,N_R)
				GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
				KeyWord = 'Equalized ' + KeyWord
			if N_R > N_W:
				RandInds = GetRandomInds(N_R,N_W)
				GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				KeyWord = 'Equalized ' + KeyWord
		lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
		lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
		if Absolute:
			WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
		else:
			WmR = GetWmR(counts_word_M, counts_ref_M)
			ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
		Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
		TimeStamp = datetimestr()+'_'
		TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
		fig, SaveTitle = PlotIt_alpha(Show, ID_pndata, TitleElts, Squares, alpha)
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0409_Data for Rebuttal\\'
		if True:
			fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
	if False: #Get Statistics
		SavePath = FigFolder+TimeStamp+SaveTitle.split('.')[0]+'.dat'
		SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
		title = GetTitleBase(TitleElts)
		fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
		fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
		fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)

##########################Calculate All Stats for FigFiles################################
if False: #Load file and calculate 2D Kolmogorov-Smirnov
	AnalysisTimeStamp = datetimestr()+'_'
	SelectedFileTokens = ['Null_vb Tweets', 'Spanish vs English Users', 'Null_loslas Tweets',
		  'MaxList Users', 'MaxList Tweets', 'Local vs Foreign Users', 'Formality6 Tweets',
		  'Spanish vs English Tweets', 'TangoFutbol Tweets', 'Stadiums Tweets', 'BocaPalermo Tweets']
	MainFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
	DataFolder = '2022_0604_Statistical comparison of cases\\'
	LoadFolders = ['01_Exact CABA', '02_Buenos Aires', '03_Equalized, Exact CABA', '04_Equalized, Buenos Aires']
	for Folder in LoadFolders:
		LoadFolderPath = MainFolder + DataFolder + Folder + '\\'
		FigFiles = os.listdir(LoadFolderPath)
		DatFiles =  [f for f in FigFiles if '.dat' in f]
		print('Number of data files: {}'.format(len(DatFiles)))
		R_Linregress, P_Linregress = [], []
		RBars, TBars = [], []
		MI1, MI2, MI3 = [], [], []
		KS_Stats,KS_Ps,SumRs,SumTs = [], [], [], []
		PlotColors = ['Black', 'Red', 'Blue', 'Green']
		PlotLineTypes = ['-', '--', '.-', '...']
		i = 0
		for Token in SelectedFileTokens:
			for f in DatFiles: #find filename associated with token
				if Token in f:
					FileName = f
			DatFilePath = LoadFolderPath + FileName
			print('File name: {}'.format(FileName))
			Data = LoadMatrixData(DatFilePath)
			R, T = Data[2], Data[3]
			#[lons_word_M, lats_word_M, counts_ref_M, counts_word_M] = Data
			#FileInfo = GetFileInfo(FileName)
			#[KeyWord, UserType, TimeFrame, MinDigits, ExtentName, NumBins, Absolute, Squares, FileTimeStamp, FileTitle, FileExt] = FileInfo
			if True: #Get Statistics
				m,b,r,p,stderr = linregress(R,T)
				R_Linregress.append(r)
				P_Linregress.append(p)
			if True: #Calculate Moran's I statistic
				RBar, TBar = np.mean(R), np.mean(T)
				N = len(R)
				D_values = [T[i]/TBar - R[i]/RBar for i in range(N)]
				Layers = 3
				MI_Stats = []
				ExcludeEmptyBins = True
				for L in range(1,Layers+1,1):
					t_0 = datetime.now()
					print('Layer number {}'.format(L))
					if ExcludeEmptyBins:
						MI = MoranI_NonZero(D_values, L)
					else:
						MI = MoranI(D_values, L, "Circle")
					MI_Stats.append(MI)
					print(MI)
					t_1 = datetime.now()
					d_t = (t_1 - t_0).seconds
					print('Calculation time for Layer {}: {} seconds.'.format(L,d_t))
				MI1.append(MI_Stats[0][0])
				MI2.append(MI_Stats[1][0])
				MI3.append(MI_Stats[2][0])
				RBars.append(RBar)
				TBars.append(TBar)
			if True: #perform Kolmogorov-Smirnov analysis
				KS_Stat, KS_P = TwoSample2DKolmogorovSmirnov(R, T, Equalize=False,Plot=False)
				KS_Stats.append(KS_Stat)
				KS_Ps.append(KS_P)
				SumR, SumT = sum(R), sum(T)
				SumRs.append(SumR)
				SumTs.append(SumT)
		ListOfListsData = [SelectedFileTokens, R_Linregress, P_Linregress, RBars, TBars, MI1, MI2, MI3, KS_Stats, KS_Ps, SumRs, SumTs]
		ListOfTitles = ["Token", "R_Linregress", "P_Linregress", "RBar", "TBar", "MI1", "MI2", "MI3", "KS_Stats", "KS_Ps", "SumRs", "SumTs"]
		SaveFolder = '05_Analysis\\'
		SaveFileName = AnalysisTimeStamp + Folder + '_StatAnalysis.txt'
		ListOfListsData_Path = MainFolder + DataFolder + SaveFolder + SaveFileName
		SaveListOfListsData(ListOfListsData_Path, ListOfListsData, ListOfTitles)
		plt.plot(R_Linregress, KS_Stats, PlotLineTypes[i], c = PlotColors[i])
		i+=1
	plt.xlabel('Linear Regression Value')
	plt.ylabel('Kolmogorov-Smirnov Statistic')
	Title = 'Comparison of KS-Statistic to Linear Regression'
	plt.title(Title)
	plt.show()
	PlotFileName = AnalysisTimeStamp + Title + '.png'
	SavePath = MainFolder + DataFolder + SaveFolder + PlotFileName
	plt.savefig(fname = SavePath, dpi=600)
	
###############################Summarizing Statistics########################
if False: #Plot Summary of Statistics
	import matplotlib.pyplot as plt
	import os
	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	plt.rcParams.update({'font.family':'Arial'})
	AnalysisTimeStamp = datetimestr()+'_'
	BasePath = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
	Folder = '2022_0604_Statistical comparison of cases\\05_Analysis\\'
	AnalysisFiles = os.listdir(BasePath + Folder)
	AllData = [LoadListOfListsData_Floats(BasePath + Folder + FileName) for FileName in AnalysisFiles if '.txt' in FileName]
	labels = ['CABA, all data', 'CABA+, all data', 'CABA, equalized', 'CABA+, equalized']
	linestyles = ['-', '--', '-.', ':']
	colors = ['k','b','r','g']
	#print(AllData[0][1])
	GraphFormat = {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}
	if True: #Kolmogorov-Smirnov statistic to linear regression
		fig1, ax1 = plt.subplots(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1]))
		for i in range(len(AllData)):
			RegP,KS = [list(tuple) for tuple in zip(*sorted(zip(AllData[i][0][1], AllData[i][0][8])))]
			#plt.plot(RegP,KS, linestyles[i], c = colors[i], label = labels[i])
			plt.plot([1-x for x in RegP],KS, linestyles[i], c = colors[i], label = labels[i])
		plt.legend(loc="upper left")
		ax1.xaxis.set_tick_params(labelsize= AxesFontSize)
		ax1.yaxis.set_tick_params(labelsize= AxesFontSize)
		#ax1.set_xlabel('Linear Regression Value', loc='center', fontsize = AxesFontSize)
		ax1.set_xlabel('1.0 - Pearson Coefficient', loc='center', fontsize = AxesFontSize)
		ax1.set_ylabel('Kolmogorov-Smirnov Statistic', loc='center', fontsize = AxesFontSize)
		title = 'Comparison of KS-test to Linear Regression'
		ax1.set_title(title,fontsize=TitleFontSize)
		if GraphFormat['tight']:
			plt.tight_layout()
		plt.show()
		SaveFile1 = AnalysisTimeStamp + title + '.pdf'
		SavePath1 = BasePath + Folder + SaveFile1
		fig1.savefig(fname = SavePath1, dpi=600, transparent = True)
	if False: #compare Moran's I vs Kolmogorov-Smirnov statistic
		fig2, ax2 = plt.subplots(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1]))
		for i in range(len(AllData)):
			KS,MI = [list(tuple) for tuple in zip(*sorted(zip(AllData[i][0][8], AllData[i][0][5])))]
			plt.plot(KS,MI, linestyles[i], c = colors[i], label = labels[i])
		plt.legend(loc="upper left")
		ax2.xaxis.set_tick_params(labelsize= AxesFontSize)
		ax2.yaxis.set_tick_params(labelsize= AxesFontSize)
		ax2.set_xlabel('Kolmogorov-Smirnov Statistic', loc='center', fontsize = AxesFontSize)
		ax2.set_ylabel('Moran\'s I', loc='center', fontsize = AxesFontSize)
		title = 'Degree of clustering vs KS-Test'
		ax2.set_title(title,fontsize=TitleFontSize)
		plt.show()
		SaveFile2 = AnalysisTimeStamp + title + '.pdf'
		SavePath2 = BasePath + Folder + SaveFile2
		fig2.savefig(fname = SavePath2, dpi=600, transparent = True)
		
############################Get Coordinates from KML file from Google Maps##############################
if False: #Plot context 1 vs context 2 using KML files from Google Maps
	if False: #load packages
		import pykml
		from pykml import parser
		import lxml
		from lxml import etree
		from lxml import objectify
	if True: #setup directories
		KML_BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
		KML_SubDir = 'PLOS one rebuttal\\2022_0518_Formal Informal Contexts\\Google Maps Coordinates\\'
		KML_Files = os.listdir(KML_BaseDir + KML_SubDir)
		KML_File_Reference, KML_File_Target = 'Formal Context.kml', 'Informal context.kml'
	if True: #print to screen
		Contexts = {'Target':{'path':KML_BaseDir + KML_SubDir + KML_File_Target}, 'Reference':{'path':KML_BaseDir + KML_SubDir + KML_File_Reference}}
		Contexts['Target']['Locations'], Contexts['Reference']['Locations'] = {}, {}
		for x in Contexts:
			kml_path = Contexts[x]['path']
			doc=None
			with open(kml_path, 'r', encoding="UTF-8") as f:
				doc = parser.parse(f).getroot()
			for fld in doc.Document.Folder:
				FolderName = fld.name
				Contexts[x]['Locations'][FolderName] = {}
				Lons, Lats = [], []
				print()
				print('Folder: {}'.format(FolderName))
				for e in fld.Placemark:
					Name = e.name
					coord = e.Point.coordinates.text.split(',')
					print('The coordinates of {} are {},{}'.format(Name, float(coord[0]), float(coord[1])))
					Lons.append(float(coord[0]))
					Lats.append(float(coord[1]))
				Contexts[x]['Locations'][FolderName]['Lons'] = Lons
				Contexts[x]['Locations'][FolderName]['Lats'] = Lats
		Target_Locations = Contexts['Target']['Locations']
		Target_Location_Types = [x for x in Target_Locations]
		Reference_Locations = Contexts['Reference']['Locations']
		Reference_Location_Types = [x for x in Reference_Locations]
	if True: #MatrixBin and plot posneg
		ExtentName = 'Exact CABA'
		extent = GetExtent(ExtentName)
		alpha = 0.4
		Show = True
		circ_scale = 1
		title = 'KMA'
		Data = [[],[],[],[],[],[]]
		for x in Target_Locations:
			Data[0] = Data[0] + Target_Locations[x]['Lons']
			Data[1] = Data[1] + Target_Locations[x]['Lats']
			Data[2] = Data[2] + [1 for y in Target_Locations[x]['Lons']] #freq
		for x in Reference_Locations:
			Data[3] = Data[3] + Reference_Locations[x]['Lons']
			Data[4] = Data[4] + Reference_Locations[x]['Lats']
			Data[5] = Data[5] + [1 for y in Reference_Locations[x]['Lons']] #freq
	if True: #Filter out repetitions, using coordinates, but considering the coordinates are not exact
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7], 'tight': False, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		DataSet = [[],[],[],[],[],[]]
		Thresh = 0.001 #Threshold for considering two elements to be the same point
		for i in range(len(Data[0])):
			LonDiff = [abs(Data[0][i] - x) for x in DataSet[0]]
			LatDiff = [abs(Data[1][i] - x) for x in DataSet[1]]
			Equiv = [(LonDiff[i] <= Thresh and LatDiff[i] <= Thresh) for i in range(len(LonDiff))]
			if not any(Equiv):
				DataSet[0].append(Data[0][i])
				DataSet[1].append(Data[1][i])
				DataSet[2].append(1)
		for i in range(len(Data[3])):
			LonDiff = [abs(Data[3][i] - x) for x in DataSet[3]]
			LatDiff = [abs(Data[4][i] - x) for x in DataSet[4]]
			Equiv = [(LonDiff[i] <= Thresh and LatDiff[i] <= Thresh) for i in range(len(LonDiff))]
			if not any(Equiv):
				DataSet[3].append(Data[3][i])
				DataSet[4].append(Data[4][i])
				DataSet[5].append(1)
		Target_Reference_DataSet = DataSet
		posnegdata = Target_Reference_DataSet
		GraphFormat = GraphFormats[ExtentName]
		pltposneg_format(GraphFormat, Show, posnegdata, extent, circ_scale = 100, title = 'KMA', img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
	if True: #plot and save figure
		BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
		SubDir = 'PLOS one rebuttal\\2022_0518_Formal Informal Contexts\\Saved Maps\\'
		SaveFolder = BaseDir + SubDir
		TimeStamp = datetimestr()+'_'
		ExtentName = 'Buenos Aires'
		extent = GetExtent(ExtentName)
		if ExtentName == 'Buenos Aires':
			Region = 'CABA'
		elif ExtentName == 'Main Buenos Aires':
			Region = 'MainBA'
		elif ExtentName == 'Exact CABA':
			Region = 'Exact CABA'
		alpha = 0.4
		Show = True
		circ_scale = 20
		TimeStamp = datetimestr()+'_'
		BaseTitle = 'Formal vs Informal from Google Maps'
		TitleExt = '_' + Region + '_Scale{}'.format(circ_scale)
		title = TimeStamp + BaseTitle + TitleExt
		GraphFormat = GraphFormats[ExtentName]
		#Plot & Save without CABA outline
		fig = pltposneg_format(GraphFormat, True, posnegdata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
		fig.savefig(fname = SaveFolder + title + '.png', dpi=600)
		#Plot & Save with CABA outline
		fig = pltposneg_format(GraphFormat, False, posnegdata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
		plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
		plt.show()
		fig.savefig(fname = SaveFolder + title + '_CabaBorder' + '.png', dpi=600)
	if False: #print to file
		for fld in doc.Document.Folder:
			FolderName = fld.name
			FilePath = KML_Dir + FolderName + '_Coordinates_01.txt'
			f = open(FilePath, 'w', encoding = 'utf-8')
			print('Location Name', 'Longitude', 'Latitude',  sep='\t', file=f)
			for e in fld.Placemark:
				PointName = e.name
				coord = e.Point.coordinates.text.split(',')
				print(PointName, float(coord[0]), float(coord[1]), sep='\t', file=f)
			f.close()

##############################Plot Single Context#####################################################
if False: #Plot single context map using KML file from Google Maps
	if False: #load packages
		import pykml
		from pykml import parser
		import lxml
		from lxml import etree
		from lxml import objectify
	if False: #setup directories
		KML_BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
		KML_SubDir = 'PLOS one rebuttal\\2022_0518_Foreign Context\\Google Maps Coordinates\\'
		KML_Files = os.listdir(KML_BaseDir + KML_SubDir)
		KML_File_Reference, KML_File_Target = 'Foreign Context.kml', 'Foreign Context.kml'
	if False: #print to screen
		Contexts = {'Target':{'path':KML_BaseDir + KML_SubDir + KML_File_Target}, 'Reference':{'path':KML_BaseDir + KML_SubDir + KML_File_Reference}}
		Contexts['Target']['Locations'], Contexts['Reference']['Locations'] = {}, {}
		for x in Contexts:
			kml_path = Contexts[x]['path']
			doc=None
			with open(kml_path, 'r', encoding="UTF-8") as f:
				doc = parser.parse(f).getroot()
			for fld in doc.Document.Folder:
				FolderName = fld.name
				Contexts[x]['Locations'][FolderName] = {}
				Lons, Lats = [], []
				print()
				print('Folder: {}'.format(FolderName))
				for e in fld.Placemark:
					Name = e.name
					coord = e.Point.coordinates.text.split(',')
					print('The coordinates of {} are {},{}'.format(Name, float(coord[0]), float(coord[1])))
					Lons.append(float(coord[0]))
					Lats.append(float(coord[1]))
				Contexts[x]['Locations'][FolderName]['Lons'] = Lons
				Contexts[x]['Locations'][FolderName]['Lats'] = Lats
		Target_Locations = Contexts['Target']['Locations']
		Target_Location_Types = [x for x in Target_Locations]
		Reference_Locations = Contexts['Reference']['Locations']
		Reference_Location_Types = [x for x in Reference_Locations]
	if False: #MatrixBin and plot posneg
		ExtentName = 'Exact CABA'
		extent = GetExtent(ExtentName)
		alpha = 0.4
		Show = True
		circ_scale = 1
		title = 'KMA'
		Data = [[],[],[],[],[],[]]
		for x in Reference_Locations:
			Data[3] = Data[3] + Reference_Locations[x]['Lons']
			Data[4] = Data[4] + Reference_Locations[x]['Lats']
			Data[5] = Data[5] + [1 for y in Reference_Locations[x]['Lons']] #freq
	if False: #Filter out repetitions, using coordinates, but considering the coordinates are not exact
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7], 'tight': False, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		DataSet = [[],[],[],[],[],[]]
		Thresh = 0.001 #Threshold for considering two elements to be the same point
		for i in range(len(Data[3])):
			LonDiff = [abs(Data[3][i] - x) for x in DataSet[3]]
			LatDiff = [abs(Data[4][i] - x) for x in DataSet[4]]
			Equiv = [(LonDiff[i] <= Thresh and LatDiff[i] <= Thresh) for i in range(len(LonDiff))]
			if not any(Equiv):
				DataSet[3].append(Data[3][i])
				DataSet[4].append(Data[4][i])
				DataSet[5].append(1)
		Target_Reference_DataSet = DataSet
		posnegdata = Target_Reference_DataSet
		GraphFormat = GraphFormats[ExtentName]
		pltposneg_format(GraphFormat, Show, posnegdata, extent, circ_scale = 100, title = 'KMA', img='y', bdr='y', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
	if True: #plot and save figure
		BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
		SubDir = 'PLOS one rebuttal\\2022_0518_Foreign Context\\Saved Maps\\'
		SaveFolder = BaseDir + SubDir
		TimeStamp = datetimestr()+'_'
		ExtentName = 'Buenos Aires'
		extent = GetExtent(ExtentName)
		if ExtentName == 'Buenos Aires':
			Region = 'CABA'
		elif ExtentName == 'Main Buenos Aires':
			Region = 'MainBA'
		elif ExtentName == 'Exact CABA':
			Region = 'Exact CABA'
		alpha = 0.4
		Show = True
		TimeStamp = datetimestr()+'_'
		BaseTitle = 'Attractions, Museums and Monuments_Google Maps'
		Scales = [10, 20, 30, 40, 50, 70]
		for Scale in Scales:
			circ_scale = Scale
			TitleExt = '_' + Region + '_Scale{}'.format(circ_scale)
			title = TimeStamp + BaseTitle + TitleExt
			GraphFormat = GraphFormats[ExtentName]
			#Plot & Save without CABA outline
			fig = pltposneg_format(GraphFormat, True, posnegdata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
			fig.savefig(fname = SaveFolder + title + '.png', dpi=600)
			#Plot & Save with CABA outline
			fig = pltposneg_format(GraphFormat, False, posnegdata, extent, circ_scale, title, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
			plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
			plt.show()
			fig.savefig(fname = SaveFolder + title + '_CabaBorder' + '.png', dpi=600)

###################################Plot Boundaries of CABA Districts###############################
def CABAMapColors(Style):
	if Style == 'Counties':
		C = [[] for i in range(15)]
		C[0] = [45,42,41,11,19,40,47]
		C[1] = [44]
		C[2] = [24,12]
		C[3] = [32,16,29,47]
		C[4] = [4,13]
		C[5] = [5]
		C[6] = [9,31]
		C[7] = [18,35,34]
		C[8] = [17,30,38]
		C[9] = [8,39,7,14,15,10]
		C[10] = [37,3,6,25]
		C[11] = [20,21,22,36]
		C[12] = [23,43,46]
		C[13] = [33]
		C[14] = [26,27,28,0,1,2]
		RegionIndices = C
		RegionColors = ['darksalmon',
		'magenta',
		'maroon',
		'gainsboro',
		'olive',
		'black',
		'mediumpurple',
		'cornflowerblue',
		'chocolate',
		'springgreen',
		'forestgreen',
		'blue',
		'yellow',
		'dimgray',
		'greenyellow']
	return RegionIndices, RegionColors
    
def GetCABABarrioIndsByCounty():
	C = [[] for i in range(15)]
	C[0] = [45,42,41,11,19,40,47]
	C[1] = [44]
	C[2] = [24,12]
	C[3] = [32,16,29,47]
	C[4] = [4,13]
	C[5] = [5]
	C[6] = [9,31]
	C[7] = [18,35,34]
	C[8] = [17,30,38]
	C[9] = [8,39,7,14,15,10]
	C[10] = [37,3,6,25]
	C[11] = [20,21,22,36]
	C[12] = [23,43,46]
	C[13] = [33]
	C[14] = [26,27,28,0,1,2]
	return C
    
if False: #Plot Regions
	if True: #Load barrios oulines data
		BasePath = 'c:/Users/Nicholas/Dropbox/Personal/misc/Olga_Computer Linguistics Article 01/Buenos Aires Information/From Argentinian Govt/'
		FileName = 'barrios.csv'
		path = BasePath + FileName
		polygons = [] # list of polygons
		with open(path, 'r') as f:
			f.readline() # burn the header line
			for line in f:
				coordpart, infopart = line.split('))')[0], line.split('))')[1] #separate out the part of the line with the coordinate info
				info = [barrio,comuna,perimetro,area] = infopart.split('\n')[0].split(',')[1:5] #extract info of neighborhood, commune, perimeter and area
				jtext = '[['+coordpart.split('((')[1].replace(',','],[').replace(' ', ',')+']]' #edit into list format
				jtext = jtext.replace('(','').replace(')','') #remove any lingering parentheses
				coordlist = json.loads(jtext) #convert from text object to actual list
				lons,lats = [x[0] for x in coordlist], [x[1] for x in coordlist]
				polygons.append([lons,lats])
	if True: #Plot Barrios
		RegionIndices, RegionColors = CABAMapColors('Counties')
		fig, ax = plt.subplots()
		for i in range(len(polygons)):
			lons,lats = polygons[i][0], polygons[i][1]
			if True: #Set colors of regions
				for j in range(len(RegionIndices)):
					if i in RegionIndices[j]:
						RegionColor = RegionColors[j]
				if True: #True = fill region with color, False = outlines only
					if i == 40:
						plt.fill(lons[0:513], lats[0:513], facecolor=RegionColor, edgecolor='black', linewidth=0.5)
						plt.fill(lons[514:1008], lats[514:1008], facecolor='white', edgecolor='black', linewidth=0.5)
					else:
						plt.fill(lons, lats, facecolor=RegionColor, edgecolor='black', linewidth=0.5)
				else:
					plt.plot(lons,lats, c=RegionColor)
			else: #Let Matplotlib set colors
				if True: #True = fill region with color, False = outlines only
					if i == 40:
						plt.fill(lons[0:513], lats[0:513], edgecolor='black', linewidth=0.5)
						plt.fill(lons[514:1008], lats[514:1008], facecolor='white', edgecolor='black', linewidth=0.5)
					else:
						plt.fill(lons, lats, edgecolor='black', linewidth=0.5)
				else:
					plt.plot(lons,lats)
		ratio = 1/np.cos(((extent[2]+extent[3])/2)*np.pi/180)
		x_left, x_right = ax.get_xlim()
		y_low, y_high = ax.get_ylim()
		ax.set_aspect(ratio)
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		title = 'Districts of CABA from Argentine Govt'
		plt.title(title)
		plt.show()
	if False: #Save CABA barrios map
		BaseDir = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\'
		SubDir = 'PLOS one rebuttal\\2022_0501_CABA Districts Self Plot\\Saved Maps\\'
		SaveFolder = BaseDir + SubDir
		TimeStamp = datetimestr()+'_'
		fig.savefig(fname = SaveFolder + TimeStamp + title + '_BordersAndFill' + '.png')


########################Plot Specific CABA Barrios######################
if False: #Plot Specific Barrios
	fig, ax = plt.subplots()
	BInds = [33,47] #indices of barrios to plot
	BColors = ['blue', 'red']
	for i in range(len(BInds)): #plot particular barrios
		plt.plot(polygons[BInds[i]][0], polygons[BInds[i]][1], c = BColors[i], linewidth = 0.5)
	ratio = 1/np.cos(((extent[2]+extent[3])/2)*np.pi/180)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	ax.set_aspect(ratio)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	title = 'Palermo and La Boca'
	plt.title(title)
	plt.show()

def IsInside(v1,v2,v3):
	n1 = np.sqrt(v1[0]**2 + v1[1]**2)
	n2 = np.sqrt(v2[0]**2 + v2[1]**2)
	n3 = np.sqrt(v3[0]**2 + v3[1]**2)
	Angle12 = np.arccos( (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2) )
	Angle13 = np.arccos( (v1[0]*v3[0]+v1[1]*v3[1])/(n1*n3) )
	Angle32 = np.arccos( (v3[0]*v2[0]+v3[1]*v2[1])/(n3*n2) )
	AngleInside = (Angle13 < Angle12) and (Angle32 < Angle12)
	if AngleInside:
		m3 = v3[1]/v3[0]
		m12 = (v2[1] - v1[1])/(v2[0] - v1[0])
		b12 = (v1[1] + v2[1] - m12*(v1[0] + v2[0]))/2
		x_intersect = b12/(m3 - m12)
		Intersect = 0 <= x_intersect <= v3[0]
		Inside = not Intersect
	elif Angle13 == 0:
		Inside = n3 < n1
	elif Angle32 ==0:
		Inside = n3 < n2
	else:
		Inside = False
	return Inside
if False:
	PolyInds = [40, 40,47,47,29,32,35,34,38,8,20,46,43,43,33,44,44,45]
	Imins = [480,0,400,0,40,40,76,5,100,56,55,102,809,0,0,1045,0,0]
	Imaxs = [513,250,1339,200,295,173,186,53,120,62,95,284,916,536,550,1196,791,1560]
	EdgeLons, EdgeLats = [],[]
	for i in range(len(PolyInds)):
		lons,lats = polygons[PolyInds[i]][0], polygons[PolyInds[i]][1]
		print('Number of points: {}'.format(len(lons)))
		if True:
			Imin,Imax = Imins[i], Imaxs[i]
		else:
			Imin,Imax = 0,len(lons)-1
		EdgeLons += lons[Imin:Imax]
		EdgeLats += lats[Imin:Imax]
	plt.plot(EdgeLons,EdgeLats, 'k-')
	plt.show()

#####################Filter by tokens and Plot Variations######################################
if False: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
	if False:
		GeoData = CollectData(ID_User_Data)
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if True: #Include only Tweets with Ref & Word Tokens
		TokenListName = 'Null_loslas' # Options: 'Null_vb', 'Null_loslas', 'BocaPalermo', 'Stadiums', 'TangoFutbol', 'Formality', 'MaxList'
		WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
		GeoData_Ref = FilterByTokens(GeoData, RefTokens, Filter=True, Include=True)
		GeoData_Word = FilterByTokens(GeoData, WordTokens, Filter=True, Include=True)
	if False: #Prep for plotting
		cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
		osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
	if False: #Load Outline of CABA
		DataPath = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0501_CABA Districts Self Plot\\Saved Maps\\2022_0509_195036_Vertices of CABA Outline.dat'
		Data, Titles = LoadListOfListsData_Floats(DataPath)
		OutlineLons, OutlineLats = Data[0], Data[1]
	if True: #MatrixBin and plot posneg
		UserType = 'All '
		List_Equalize = [False for i in range(8)] + [True for i in range(8)]
		List_ExtentName = ['Exact CABA' for i in range(4)] + ['Buenos Aires' for i in range(4)]
		List_ExtentName = List_ExtentName + List_ExtentName
		List_NumBins = [75, 75, 75, 75, 100, 100, 100, 100, 75, 75, 75, 75, 100, 100, 100, 100]
		List_Squares = [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True]
		Calc = [True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False]
		List_Hold = [False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True]
		for i in range(len(List_NumBins)):
			extent = GetExtent(List_ExtentName[i])
			FilterTimeAndDigits = True
			NumBins = List_NumBins[i]
			Absolute = False
			Squares = List_Squares[i]
			RelScale = 2
			alpha = 0.4
			ElimZeros = True
			Show = not List_Hold[i]
			KeyWord = TokenListName+' Tweets'
			if List_Equalize[i]:
				KeyWord = 'Equalized ' + KeyWord
			if Calc[i]:
				GeoData_W = FilterByExtent(GeoData_Word, extent, Filter=True)
				GeoData_R = FilterByExtent(GeoData_Ref, extent, Filter=True)
				if List_Equalize[i]: #Randomly filter the larger data set to equalize the numbers
					N_W, N_R = len(GeoData_W[0]), len(GeoData_R[0])
					if N_W > N_R:
						RandInds = GetRandomInds(N_W,N_R)
						GeoData_W = [[x[i] for i in RandInds] for x in GeoData_W]
					if N_R > N_W:
						RandInds = GetRandomInds(N_R,N_W)
						GeoData_R = [[x[i] for i in RandInds] for x in GeoData_R]
				lons_ref_M, lats_ref_M, counts_ref_M, Elts_ref_M = MatrixBinElts(GeoData_R[0], GeoData_R[1], extent, NumBins, NumBins, Elts=[])
				lons_word_M, lats_word_M, counts_word_M, Elts_word_M = MatrixBinElts(GeoData_W[0], GeoData_W[1], extent, NumBins, NumBins, Elts=[])
				if Absolute:
					WmRabs = GetWmRabs(counts_word_M, counts_ref_M)
					ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmRabs, ElimZeros)
				else:
					WmR = GetWmR(counts_word_M, counts_ref_M)
					ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
			Scale = GetScale(List_ExtentName[i], List_Squares[i], Absolute, RelScale)
			TimeStamp = datetimestr()+'_'
			TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, List_NumBins[i], List_ExtentName[i], Absolute, Scale]
			fig, SaveTitle = PlotIt_alpha_bdr(Show, ID_pndata, TitleElts, List_Squares[i], alpha, border='n')
			if not Show:
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle = SaveTitle[0:-4]+'_CabaBorder'+SaveTitle[-4:]
			FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0511_Data for Rebuttal_Automated\\'
			if True:
				fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=300)
			if Calc[i]: #Get Statistics
				SavePath = FigFolder+TimeStamp+SaveTitle[0:-4]+'.dat'
				SaveMatrixData(SavePath, lons_ref_M, lats_ref_M, counts_ref_M, counts_word_M)
				title = GetTitleBase(TitleElts)
				fig,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title, FontSize = 16, CircleSize = 20)
				fig.savefig(fname = FigFolder+TimeStamp+title+'_FreqPlot.png', dpi=300)
				fig,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
				fig.savefig(fname = FigFolder+TimeStamp+title+'_NormAngHist.png', dpi=300)



##########################Plotting All Tweets###########################
if True: #Main Program
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
		GeoData = CollectData(ID_User_Data)
	if False: #Filter by digits and by time-frame
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if True: #MatrixBin and plot posneg
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0514_All Tweets Graphs\\'
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		List_ExtentName = ['Exact CABA', 'Exact CABA', 'Buenos Aires', 'Buenos Aires']
		List_NumBins = [75, 75, 100, 100]
		Calc = [True, False, True, False]
		List_Scale = [0.05, 0.05, 0.01, 0.01]
		List_Hold = [False, True, False, True]
		for i in range(len(List_NumBins)):
			extent = GetExtent(List_ExtentName[i])
			NumBins = List_NumBins[i]
			alpha = 0.4
			Show = not List_Hold[i]
			if Calc[i]:
				GeoData_A = FilterByExtent(GeoData, extent, Filter=True)
				lons_M, lats_M, counts_M, Elts_M = MatrixBinElts(GeoData_A[0], GeoData_A[1], extent, NumBins, NumBins, Elts=[])
				coord = [lons_M, lats_M]
				freq = counts_M
			Scale = List_Scale[i]
			TimeStamp = datetimestr()+'_'
			Title = 'Distribution of all tweets over extent \'{}\'_Scale{}'.format(List_ExtentName[i], Scale)
			fig = mkscatplt_show_format(GraphFormats[List_ExtentName[i]], Show, coord, freq, extent, circ_scale = Scale, title = Title, img='y', bdr='n', grd='n', clr='purple', cmap='bwr', alpha = 0.4, osm_img = osm_img)
			SaveTitle = Title + '.png'
			if not Show:
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle = SaveTitle[0:-4]+'_CabaBorder'+SaveTitle[-4:]
			if True:
				fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=600)

#######################################Plotting ArgSp vs PenSp comparison plots####################################
if False: #Plotting ArgSp vs PenSp comparison plots
	if False: #Get Data
		drive = "E:\\"
		CorpusDir = drive + 'Corpora\\ExtentBasedGeoLocationCorpora\\es\\Buenos Aires'
		tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations = GetData(CorpusDir)
	if False: #Bin Users
		ID_User_Data, BinnedElts = QuickBin(tweet_IDs, tweet_lons, tweet_lats, tweet_texts, tweet_dates, tweet_times, tweet_names, tweet_locations)
		GeoData = CollectData(ID_User_Data)
	if False: #Filter by digits and by time-frame
		MinDigits = 5 # Positive number, including 0: keep digits this number or above; Negative number: keep only digits with this number
		TimeFrame = 'AllTimes' # Options: 'Pre06.2019', 'Post06.2019', 'AllTimes'
		GeoData = FilterDigitsAndTime(GeoData, MinDigits, TimeFrame, Filter = True)
	if False: #Include only Tweets with Ref & Word Tokens
		TokenListName = 'MaxList' # Options: 'Null_vb', 'Null_loslas', 'BocaPalermo', 'Stadiums', 'TangoFutbol', 'Formality', 'MaxList'
		WordTokens, RefTokens = GetTokens(TokenListName)['words2check'], GetTokens(TokenListName)['RefWords']
		GeoData_Ref = FilterByTokens(GeoData, RefTokens, Filter=True, Include=True)
		GeoData_Word = FilterByTokens(GeoData, WordTokens, Filter=True, Include=True)
	if True: #MatrixBin and plot posneg
		FigFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\2022_0514_ArgSp vs PenSp Comparison\\'
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7], 'tight': False, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		List_ExtentName = ['Exact CABA', 'Exact CABA', 'Buenos Aires', 'Buenos Aires'] + ['Exact CABA', 'Exact CABA', 'Buenos Aires', 'Buenos Aires']
		List_NumBins = [75, 75, 100, 100] + [75, 75, 100, 100]
		Calc = [True, False, True, False] + [True, False, True, False]
		List_Scale = [2, 2, 0.5, 0.5] + [5, 5, 1, 1]
		List_Hold = [False, True, False, True] + [False, True, False, True]
		List_Color = ['red' for i in range(4)] + ['blue' for i in range(4)]
		List_Title = ['Distribution of ArgSp tweets' for i in range(4)] + ['Distribution of PenSp tweets' for i in range(4)]
		for i in range(len(List_NumBins)):
			alpha = 0.4
			Show = not List_Hold[i]
			if Calc[i]:
				extent = GetExtent(List_ExtentName[i])
				if i in [0, 1, 2, 3]:
					GeoData_A = FilterByExtent(GeoData_Word, extent, Filter=True)
				if i in [4, 5, 6, 7]:
					GeoData_A = FilterByExtent(GeoData_Ref, extent, Filter=True)
				NumBins = List_NumBins[i]
				lons_M, lats_M, counts_M, Elts_M = MatrixBinElts(GeoData_A[0], GeoData_A[1], extent, NumBins, NumBins, Elts=[])
				coord = [lons_M, lats_M]
				freq = counts_M
			Scale = List_Scale[i]
			TimeStamp = datetimestr()+'_'
			Title = List_Title[i] + ' over extent \'{}\'_Scale{}'.format(List_ExtentName[i], Scale)
			fig = mkscatplt_show_format(GraphFormats[List_ExtentName[i]], Show, coord, freq, extent, circ_scale = Scale, title = Title, img='y', bdr='n', grd='n', clr=List_Color[i], cmap='bwr', alpha = 0.4, osm_img = osm_img)
			SaveTitle = Title + '.png'
			if not Show:
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle = SaveTitle[0:-4]+'_CabaBorder'+SaveTitle[-4:]
			if True:
				fig.savefig(fname = FigFolder+TimeStamp+SaveTitle, dpi=600)


###########################Generate & Save Figures from Graph Data Files#################################
if False: #Load file and create figures
	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	plt.rcParams.update({'font.family':'Arial'})
	if True: #Get list of files
		LoadFolderMain = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
		LoadFolderSub = '2022_0605_Adobe Illustrator Compatible Figs\\'
		#LoadFolderSub = '2022_0605_Test\\'
		LoadFolder = LoadFolderMain + LoadFolderSub
		FigFiles = os.listdir(LoadFolder)
		DatFiles =  [f for f in FigFiles if '.dat' in f]
		print('Number of data files: {}'.format(len(DatFiles)))
		AnalysisTimeStamp = datetimestr()+'_'
		SaveFolder = smartdir(LoadFolder + AnalysisTimeStamp + 'Analysis') + '\\'
	if False: #Prep for plotting
		cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
		osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
	if True: #Select Specific File
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7],
						 'tight': False, 'NumTics': [5, 6], 'TitleFontSize': 16,
						 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7],
						 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		for FileName in DatFiles:
			print('File name: {}'.format(FileName))
			Data = LoadMatrixData(LoadFolder+FileName)
			[lons_word_M, lats_word_M, counts_ref_M, counts_word_M] = Data
			FileTimeStamp = FileName[0:17]
			FileTitle = FileName[17:-4]
			FileExt = FileName[-4:]
			[KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent, ScaleAndType] = FileTitle.split(',')
			BaseTitle = ','.join([KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent])
			if MinDigitsTxt[0] == 'A':
			    MinDigits = 0
			else:
			    MinDigits = float(MinDigitsTxt[0])
			if 'Exact CABA' in BinningAndExtent:
			    ExtentName = 'Exact CABA'
			    NumBins = float(BinningAndExtent.split('Exact CABA')[0].split('Bin')[1])
			elif 'CABA' in BinningAndExtent:
			    ExtentName = 'Buenos Aires'
			    NumBins = float(BinningAndExtent.split('CABA')[0].split('Bin')[1])
			RelScale = 2
			Squares = False
			if True: #Get Information
				GraphFormat = GraphFormats[ExtentName]
				extent = GetExtent(ExtentName)
				alpha = 0.4
				ElimZeros = True
				#Show = True
				WmR = GetWmR(counts_word_M, counts_ref_M)
				ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
				Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
				TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
			if False: #Create Plots of distributions
				Title_M_Sqr = BaseTitle
				#Plot Squares
				fig_M_Sqr = pltposneg_square_format_fill(GraphFormat, True, NumBins, ID_pndata, extent,
					title = Title_M_Sqr, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				SaveTitle_M_Sqr = Title_M_Sqr + ', Squares.png'
				#Plot Squares with CABA Border
				fig_M_Sqr_CABA = pltposneg_square_format_fill(GraphFormat, False, NumBins, ID_pndata, extent,
					title = Title_M_Sqr, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Sqr_CABA = Title_M_Sqr + ', Squares_CabaBorder.png'
				#Plot Circles
				Title_M_Circ = BaseTitle + ',Scale{}'.format(Scale)
				fig_M_Circ = pltposneg_format(GraphFormat, True, ID_pndata, extent, circ_scale = Scale, title = Title_M_Circ,
					img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				SaveTitle_M_Circ = Title_M_Circ + ', Circles.png'
				#Plot Circles with CABA Border
				fig_M_Circ_CABA = pltposneg_format(GraphFormat, False, ID_pndata, extent, circ_scale = Scale, title = Title_M_Circ,
					img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Circ_CABA = Title_M_Circ + ', Circles_CabaBorder.png'
			if True: #Get Statistics Figures
				title = GetTitleBase(TitleElts)
				fig_FP,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True, title=title,
					FontSize = 16, CircleSize = 20)
				fig_FP_Zoom,LinReg = FreqPlot2(counts_ref_M, counts_word_M, fit=True, avgline=True, staterr=True,
					title=title+' Zoom', FontSize = 16, CircleSize = 20)
				fig_NAH,stddev = AngleHist_Norm2(counts_ref_M, counts_word_M, title = title, avgline=True, FontSize = 16, CircleSize = 20)
			if True: #Save Figures
				#fig_M_Sqr.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Sqr, dpi=600)
				#fig_M_Sqr_CABA.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Sqr_CABA, dpi=600)
				#fig_M_Circ.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Circ, dpi=600)
				#fig_M_Circ_CABA.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Circ_CABA, dpi=600)
				#fig_FP.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_FreqPlot.png', dpi=600)
				#fig_FP_Zoom.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_FreqPlot_Zoom.png', dpi=600)
				#fig_NAH.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_NormAngHist.png', dpi=600)
				fig_FP.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_FreqPlot.pdf', dpi=600, transparent = True)
				fig_FP_Zoom.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_FreqPlot_Zoom.pdf', dpi=600, transparent = True)
				fig_NAH.savefig(fname = SaveFolder+FileTimeStamp+BaseTitle+'_NormAngHist.pdf', dpi=600, transparent = True)

###############################Generate & save Differential Distribution figures from Graph Data with Barrios Overlays########################
if False: #Create figures from Graph Data with barrio overlays
	if True: #Get list of files
		LoadFolderMain = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
		LoadFolderSub = '2022_0620_Overlay barrios for local vs foreign\\'
		LoadFolder = LoadFolderMain + LoadFolderSub
		FigFiles = os.listdir(LoadFolder)
		DatFiles =  [f for f in FigFiles if '.dat' in f]
		print('Number of data files: {}'.format(len(DatFiles)))
		AnalysisTimeStamp = datetimestr()+'_'
		SaveFolder = smartdir(LoadFolder + AnalysisTimeStamp + 'Analysis') + '\\'
	if False: #Prep for plotting
		cimgt.Stamen.get_image = image_spoof # reformat web request for street map spoofing
		osm_img = cimgt.Stamen('terrain') # spoofed, downloaded street map
	if True: #Get indices of Barrios
		BARRIO_IND = {'Chacarita': 0, 'Paternal': 1, 'Villa Crespo': 2, 'Villa Del Parque': 3, 'Almagro': 4, 'Caballito': 5,
		 'Villa Santa Rita': 6, 'Monte Castro': 7, 'Villa Real': 8, 'Flores': 9, 'Floresta': 10, 'Constitucion': 11,
		 'San Cristobal': 12, 'Boedo': 13, 'Velez Sarsfield': 14, 'Villa Luro': 15, 'Parque Patricios': 16,
		 'Mataderos': 17, 'Villa Lugano': 18, 'San Telmo': 19, 'Saavedra': 20, 'Coghlan': 21, 'Villa Urquiza': 22,
		 'Colegiales': 23, 'Balvanera': 24, 'Villa General Mitre': 25, 'Parque Chas': 26, 'Agronomia': 27,
		 'Villa Ortuzar': 28, 'Barracas': 29, 'Parque Avellaneda': 30, 'Parque Chacabuco': 31, 'Nueva Pompeya': 32,
		 'Palermo': 33, 'Villa Riachuelo': 34, 'Villa Soldati': 35, 'Villa Pueyrredon': 36, 'Villa Devoto': 37,
		 'Liniers': 38, 'Versalles': 39, 'Puerto Madero': 40, 'Monserrat': 41, 'San Nicolas': 42, 'Belgrano': 43,
		 'Recolleta': 44, 'Retiro': 45, 'Nunez': 46, 'La Boca': 47}
	if True: #Load Data from Files
		GraphFormats = {'Buenos Aires': {'FigSize': [10,7], 'tight': False, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}, 'Exact CABA': {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}}
		#BInds = [33,47] # Indices of Barrios to overlay
		Barrios_to_Overlay = ['Palermo', 'Recolleta', 'Retiro', 'San Nicolas', 'Monserrat', 'Puerto Madero', 'San Telmo', 'La Boca']
		BInds = [BARRIO_IND[x] for x in Barrios_to_Overlay] # Indices of Barrios to overlay
		#BColors = ['blue', 'red'] #Overlay colors
		BColors = ['black' for x in BInds] #Overlay colors
		for FileName in DatFiles:
			print('File name: {}'.format(FileName))
			Data = LoadMatrixData(LoadFolder+FileName)
			[lons_word_M, lats_word_M, counts_ref_M, counts_word_M] = Data
			FileTimeStamp = FileName[0:17]
			FileTitle = FileName[17:-4]
			FileExt = FileName[-4:]
			[KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent, ScaleAndType] = FileTitle.split(',')
			BaseTitle = ','.join([KeyWord, UserType, TimeFrame, MinDigitsTxt, BinningAndExtent])
			if MinDigitsTxt[0] == 'A':
			    MinDigits = 0
			else:
			    MinDigits = float(MinDigitsTxt[0])
			if 'Exact CABA' in BinningAndExtent:
			    ExtentName = 'Exact CABA'
			    NumBins = float(BinningAndExtent.split('Exact CABA')[0].split('Bin')[1])
			elif 'CABA' in BinningAndExtent:
			    ExtentName = 'Buenos Aires'
			    NumBins = float(BinningAndExtent.split('CABA')[0].split('Bin')[1])
			RelScale = 2
			Squares = False
			if True: #MatrixBin and plot posneg
				GraphFormat = GraphFormats[ExtentName]
				extent = GetExtent(ExtentName)
				alpha = 0.4
				ElimZeros = True
				#Show = True
				WmR = GetWmR(counts_word_M, counts_ref_M)
				ID_pndata = GetPosNegData(lons_word_M, lats_word_M, WmR, ElimZeros)
				Scale = GetScale(ExtentName, Squares, Absolute, RelScale)
				TitleElts = [KeyWord, UserType, TimeFrame, MinDigits, NumBins, ExtentName, Absolute, Scale]
				Title_M_Sqr = BaseTitle
				#Plot Squares with Barrios
				fig_M_Sqr_Barr = pltposneg_square_format_fill(GraphFormat, False, NumBins, ID_pndata, extent,
					title = Title_M_Sqr, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				for i in range(len(BInds)): #plot particular barrios
					plt.plot(polygons[BInds[i]][0], polygons[BInds[i]][1], c = BColors[i], linewidth = 1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Sqr_Barr = Title_M_Sqr + ', Squares_Barrios.png'
				#Plot Squares with Barrios & CABA Border
				fig_M_Sqr_Barr_CABA = pltposneg_square_format_fill(GraphFormat, False, NumBins, ID_pndata, extent,
					title = Title_M_Sqr, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				for i in range(len(BInds)): #plot particular barrios
					plt.plot(polygons[BInds[i]][0], polygons[BInds[i]][1], c = BColors[i], linewidth = 1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Sqr_Barr_CABA = Title_M_Sqr + ', Squares_Barrios_CabaBorder.png'
				#Plot Circles with Barrios
				Title_M_Circ = BaseTitle + ',Scale{}'.format(Scale)
				fig_M_Circ_Barr = pltposneg_format(GraphFormat, False, ID_pndata, extent, circ_scale = Scale,
					title = Title_M_Circ, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				for i in range(len(BInds)): #plot particular barrios
					plt.plot(polygons[BInds[i]][0], polygons[BInds[i]][1], c = 'black', linewidth = 1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Circ_Barr = Title_M_Circ + ', Circles_Barrios.png'
				#Plot Circles with Barrios & CABA Border
				fig_M_Circ_Barr_CABA = pltposneg_format(GraphFormat, False, ID_pndata, extent, circ_scale = Scale,
					title = Title_M_Circ, img='y', bdr='n', grd='n', poscolor='r', negcolor='b', alpha=alpha, osm_img=osm_img)
				plt.plot(OutlineLons, OutlineLats, c = 'black', linewidth = 1, alpha=1, transform=crs.PlateCarree())
				for i in range(len(BInds)): #plot particular barrios
					plt.plot(polygons[BInds[i]][0], polygons[BInds[i]][1], c = 'black', linewidth = 1, transform=crs.PlateCarree())
				plt.show()
				SaveTitle_M_Circ_Barr_CABA = Title_M_Circ + ', Circles_Barrios_CabaBorder.png'
			if True: #Save Figures
				fig_M_Sqr_Barr.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Sqr_Barr, dpi=600)
				fig_M_Sqr_Barr_CABA.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Sqr_Barr_CABA, dpi=600)
				fig_M_Circ_Barr.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Circ_Barr, dpi=600)
				fig_M_Circ_Barr_CABA.savefig(fname = SaveFolder+FileTimeStamp+SaveTitle_M_Circ_Barr_CABA, dpi=600)
				
#######################Get Income Statistics####################
#Data sourced from: https://data.buenosaires.gob.ar/dataset/encuesta-anual-hogares/resource/3a45c563-396d-42de-ba93-8a93729e0723
if False: #get data of monthly income per person as a function of the community
        matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	plt.rcParams.update({'font.family':'Arial'})
	if False: #get data
		MainFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
		StatFolder = '2022_0523_Statistics by Community_CABA\\'
		StatFile = 'encuesta-anual-hogares-2019.csv'
		StatFilePath = MainFolder + StatFolder + StatFile
		ListOfListsData, Titles = LoadListOfListsData_Delim(DataPath = StatFilePath, Delimiter = ',')
	if False: #Isolate monthly income per capita data
		MonthlyIncomeInd = Titles.index('ingreso_per_capita_familiar')
		CommuneInd = Titles.index('comuna')
		MonthlyIncomeRaw = [float(x) for x in ListOfListsData[MonthlyIncomeInd]]
		CommuneRaw = [int(x) for x in ListOfListsData[CommuneInd]]
		Commune, MonthlyIncome = [], []
		for i in range(len(MonthlyIncomeRaw)): #eliminate zeros in monthly income data
			if MonthlyIncomeRaw[i] > 0:
				MonthlyIncome.append(MonthlyIncomeRaw[i])
				Commune.append(CommuneRaw[i])
	if False: #Bin Data by Commune
		CommuneBin = [i+1 for i in range(15)]
		MonthlyIncomeBin = [[] for i in range(15)]
		for i in range(len(MonthlyIncome)):
			Ind = Commune[i] - 1
			MonthlyIncomeBin[Ind].append(MonthlyIncome[i])
		plt.scatter(Commune, MonthlyIncome)
		plt.show()
	if False: #Average monthly income in each bin
		MonthlyIncomeBinMean = [np.mean(x) for x in MonthlyIncomeBin]
		plt.plot(CommuneBin, MonthlyIncomeBinMean)
		plt.show()
	if False: #Load barrios oulines data
		BasePath = 'c:/Users/Nicholas/Dropbox/Personal/misc/Olga_Computer Linguistics Article 01/Buenos Aires Information/From Argentinian Govt/'
		FileName = 'barrios.csv'
		path = BasePath + FileName
		polygons = [] # list of polygons
		with open(path, 'r') as f:
			f.readline() # burn the header line
			for line in f:
				coordpart, infopart = line.split('))')[0], line.split('))')[1] #separate out the part of the line with the coordinate info
				info = [barrio,comuna,perimetro,area] = infopart.split('\n')[0].split(',')[1:5] #extract info of neighborhood, commune, perimeter and area
				jtext = '[['+coordpart.split('((')[1].replace(',','],[').replace(' ', ',')+']]' #edit into list format
				jtext = jtext.replace('(','').replace(')','') #remove any lingering parentheses
				coordlist = json.loads(jtext) #convert from text object to actual list
				lons,lats = [x[0] for x in coordlist], [x[1] for x in coordlist]
				polygons.append([lons,lats])
	if True: #Plot Barrios, coloring by Monthly Income
		#RegionIndices, RegionColors = CABAMapColors('Counties')
		AnalysisTimeStamp = datetimestr()+'_'
		RegionIndices = GetCABABarrioIndsByCounty()
		Color = 'blue' #'crimson'
		MIBM_min, MIBM_max = min(MonthlyIncomeBinMean), max(MonthlyIncomeBinMean)
		Alpha_min, Alpha_max = 0.1, 1
		RegionColors = [Color for x in RegionIndices]
		RegionAlphas = [Alpha_min + (Alpha_max - Alpha_min)*(x - MIBM_min)/(MIBM_max - MIBM_min) for x in MonthlyIncomeBinMean]
		GraphFormat = {'FigSize': [7, 7], 'tight': True, 'NumTics': [5, 6], 'TitleFontSize': 16, 'AxesFontSize': 16}
		fig, ax = plt.subplots(figsize=(GraphFormat['FigSize'][0],GraphFormat['FigSize'][1]))
		for i in range(len(polygons)):
			lons,lats = polygons[i][0], polygons[i][1]
			for j in range(len(RegionIndices)):
				if i in RegionIndices[j]:
					RegionColor = RegionColors[j]
					RegionAlpha = RegionAlphas[j]
			if i == 40: #address fill problem for region of Puerto Madero and color water areas white
				plt.fill(lons[0:513], lats[0:513], facecolor=RegionColor, alpha=RegionAlpha, linewidth=0)
				plt.fill(lons[514:1008], lats[514:1008], facecolor='white', alpha=1, linewidth=0)
				plt.plot(lons,lats, c='black', linewidth = 0.5)
			else:
				plt.fill(lons, lats, facecolor=RegionColor, alpha=RegionAlpha, linewidth=0)
				plt.plot(lons,lats, c='black', linewidth = 0.5)
		ratio = 1/np.cos(((extent[2]+extent[3])/2)*np.pi/180)
		x_left, x_right = ax.get_xlim()
		y_low, y_high = ax.get_ylim()
		ax.set_aspect(ratio)
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		ax.xaxis.set_tick_params(labelsize= AxesFontSize)
		ax.yaxis.set_tick_params(labelsize= AxesFontSize)
		ax.set_xlabel('Longitude', loc='center', fontsize = AxesFontSize)
		ax.set_ylabel('Latitude', loc='center', fontsize = AxesFontSize)
		title = 'Monthly Income per Capita from Argentine Govt'
		ax.set_title(title,fontsize=TitleFontSize)
		#plt.title(title)
		if GraphFormat['tight']:
			plt.tight_layout()
		plt.show()
	if True: #Save figure
		SaveFolder = 'Monthly Income per Capita by County\\'
		SaveTitle = AnalysisTimeStamp + 'Monthly Income per Capita by County_Blue_600dpi.pdf'
		SavePath = MainFolder + StatFolder + SaveFolder + SaveTitle
		fig.savefig(fname = SavePath, dpi=600, transparent = True)



###############################Create Fancy Bar Plot comparing Signal Strength Across Cases###########################
from matplotlib.ticker import MaxNLocator

def plot_Category_Signals(Signal, Category, Label):
    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.3, right=0.95, bottom=0.18, top=0.92)
    pos = np.arange(len(Category))
    rects = ax1.barh(pos, Signal, align='center', height=0.5, color='m', tick_label=Category)
    ax1.set_title('Signal Strength by Category', fontsize = 16)
    ax1.set_xlim([-0.05, 1.05])
    ax1.xaxis.set_major_locator(MaxNLocator(6))
    ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    x_label = ax1.text(.5, -.12, '1.0 - Pearson Coefficient', horizontalalignment='center', fontsize=16, transform=ax1.transAxes)
    i=0
    offset = 0.02
    for rect in rects: # Lastly, write in the label inside each bar to aid in interpretation
        width = rect.get_width()
        # The bars aren't wide enough to print the ranking inside
        sig = Signal[i]
        if sig < 0.4:
            xloc = width + offset
            clr = 'black'
            align = 'left'
        else:
            #xloc = 0.98*width
            xloc = width - offset
            clr = 'white'
            align = 'right'
        yloc = rect.get_y() + rect.get_height()/2.0 # Center the text vertically in the bar
        label = ax1.text(xloc, yloc, Label[i], horizontalalignment=align,
                         verticalalignment='center', color=clr, weight='bold', clip_on=True)
        i+=1
    return {'fig': fig, 'ax': ax1, 'bars': rects} # return all of the artists created

if False:
	AnalysisTimeStamp = datetimestr()+'_'
	if True: #Use New Data
		P_values = [0.9905, 0.9847, 0.9838, 0.9759, 0.9414, 0.9346,
			    0.5481, 0.2872, 0.2223, 0.0012, 0.0009]
		Signal = [1-x for x in P_values]
		Category = ['Orthographic Elements', 'User Language', 'Grammatical Words',
			    'User Dialect', 'Tweet Dialect', 'User Origin', 'Language Style',
			    'Tweet Language', 'Activity Names', 'Location Names', 'Neighborhood Names']
		Label = ['v vs b Tweets', 'Spanish vs English Users', 'Los vs Las Tweets', 'ArgSp vs PenSp Users',
			 'ArgSp vs PenSp Tweets', 'Local vs Foreign Users', 'Informal vs Formal Tweets',
			 'Spanish vs English Tweets', 'tango vs fútbol Tweets',
			 '"Monumental" vs Bombonera Tweets', 'La Boca vs Palermo Tweets']
		##Null v vs b Tweets:			p = 0.9905
		##Spanish vs English Users:		p = 0.9847
		##Null los vs las Tweets: 		p = 0.9838
		##ArgSp vs PenSp Users:			p = 0.9759
		##ArgSp vs PenSp Tweets:		p = 0.9414
		##Local vs Foreign Users: 		p = 0.9346
		##Informal vs Formal Tweets:		p = 0.5481
		##Spanish vs English Tweets:		p = 0.2872
		##Tango vs Futbol Tweets:		p = 0.2223
		##“Monumental” vs Bombonera Tweets:	p = 0.0012
		##Boca vs Palermo Tweets:		p = 0.0009
	else: #Use Old Data
		Signal = [0.009000000000000008, 0.016000000000000014, 0.02300000000000002, 0.06399999999999995,
			  0.485, 0.698, 0.78, 0.999048, 0.999144]
		Category = ['Orthographic Elements', 'Grammatical Words', 'User Dialect', 'User Origin',
			    'Language Style', 'User Language', 'Activity Names', 'Location Names', 'Neighborhood Names']
		Label = ['v vs b Tweets', 'Los vs Las Tweets', 'ArgSp vs PenSp Users', 'Local vs Foreign Users',
			 'Informal vs Formal Tweets', 'Spanish vs English Users', 'tango vs fútbol Tweets',
			 '"Monumental" vs Bombonera Tweets', 'La Boca vs Palermo Tweets']
	arts = plot_Category_Signals(Signal, Category, Label)
	plt.show()
	MainFolder = 'C:\\Users\\Nicholas\\Dropbox\\Personal\\misc\\Olga_Computer Linguistics Article 01\\PLOS one rebuttal\\'
	SaveFolder = '2022_0604_Fancy Bar Graph of  Pearson Coeff\\'
	SaveTitle = AnalysisTimeStamp + 'Comparison of Regression Values Across Cases_new.png'
	SavePath = MainFolder + SaveFolder + SaveTitle
	arts['fig'].savefig(fname = SavePath, dpi=600)
