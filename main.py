import time
import math
start_time=time.clock()
import csv
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.linear_model import LinearRegression 
from collections import Iterable

def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, basestring):
             for x in flatten(item):
                 yield x
         else:        
             yield item

vectorizer = CountVectorizer(min_df=1)

ins=open("\train.csv", "rb") 
sheet=csv.reader(ins)

columns=defaultdict(list)
columns1=defaultdict(list)

with open('\Train.csv') as f:
   reader=csv.DictReader(f)
   for row in reader:
      for (k,v) in row.items():
         columns[k].append(v)

n=77946
with open('\Test.csv') as f:
   reader=csv.DictReader(f)
   for row in reader:
      for (k,v) in row.items():
         columns1[k].append(v)

train_tweet=columns['tweet'][:n]
test_tweet=columns1['tweet'][:n]

for row in range(77946):
    columns['tweet'][row]=re.sub(r'[#]','', columns['tweet'][row])
    columns['tweet'][row]=''.join(columns['tweet'][row]+' '+columns['state'][row])
for row in range(42157):
    columns1['tweet'][row]=re.sub(r'[#]','', columns1['tweet'][row])
    columns1['tweet'][row]=''.join(columns1['tweet'][row]+' '+columns1['state'][row])
#print columns['tweet'][2]    
#print columns1['tweet'][2]

count_vectorizer = CountVectorizer()
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

weather=['land breeze', 'advisory', 'atmosphere', 'anemometer', 'humid', 'storm surge', 'easterlies', 'air pressure', 'fall', 'scattered', 'nimbostratus', 'gully washer', 'evaporation', 'sleet', 'radar', 'thermal', 'ice', 'twilight', 'cold', 'stratocumulus', 'macroburst', 'current', 'summer', 'disturbance', 'low ', 'zone', 'cumulus', 'isobar', 'stratus', 'leeward','sky', 'isobar ', 'isotherm', 'snow line', 'ice age', 'pollutant', 'cloud bank', 'duststorm', 'wind vane', 'contrail', 'hygrometer', 'downpour', 'sea breeze', 'air', 'aurora', 'ice storm', 'watch', 'cumuliform', 'cirriform', 'Air', 'NEXRAD', 'ground fog', 'supercell', 'trough', 'knot', 'compass', 'front', 'turbulence', 'cumulonimbus', 'precipitation', 'halo', 'vapour', 'gustnado', 'inversion', 'funnel cloud', 'gust', 'icicle', 'avalanche', 'rain shadow ', 'humidity', 'ozone', 'vortex', 'upwind', 'microburst', 'whiteout', 'wind shear', 'monsoon', 'troposphere', 'vortex ', 'snowstorm', 'cirrus', 'whirlwind', 'altocumulus', 'eye wall ', 'tropical storm', 'fair', 'tornado alley', 'landfall', 'super cell', 'westerlies', 'flood stage', 'hail', 'freeze', 'typhoon ', 'wind chill', 'partly cloudy', 'downwind', 'cloudy', 'hygrometer ', 'wind chill factor', 'air mass', 'graupel', 'lightning', 'drizzle', 'depression', 'tropical depression', 'sandstorm', 'gust ', 'thunder', 'low clouds', 'updraft', 'hurricane', 'balmy', 'forecast', 'cell', 'haze', 'fog', 'sunrise', 'landspout', 'cyclone', 'ozone ', 'smog', 'prevailing wind', 'orographic cloud', 'prevailing wind ', 'lake effect', 'barometer', 'low pressure system', 'degree', 'spring', 'slush', 'National Weather Service (NWC)', 'dew point', 'squall', 'pressure', 'Tropic of Cancer', 'rain', 'water cycle', 'emergency radio', 'squall line', 'jet stream', 'doldrums', 'water', 'climate', 'St. Elmos fire', 'hydrology', 'greenhouse effect', 'biosphere', 'Fujita scale', 'vapor trail', 'mammatus cloud', 'steam', 'smoke', 'ridge', 'parhelion', 'permafrost', 'snow level', 'drought ', 'wedge', 'outlook', 'blizzard', 'snowsquall', 'heat wave', 'upwelling', 'feeder bands', 'snowfall', 'windsock', 'weather', 'heat index', 'monsoon ', 'jet stream ','weather vane', 'waterspout', 'mist', 'cloud', 'sun dog', 'funnel cloud ', 'eye', 'temperature', 'cold snap', 'atmospheric pressure', 'Kelvin', 'updraft ', 'earthlight', 'tropical', 'eddy', 'atmosphere ', 'calm', 'storm', 'altostratus', 'breeze ', 'twister', 'low', 'dry', 'hydrosphere', 'freezing rain', 'thaw', 'moisture', 'nimbus', 'swell', 'frost', 'snow shower', 'dry ', 'visibility', 'meteorologist', 'firewhirl', 'heat', 'knot ', 'typhoon', 'convergence', 'wall cloud', 'mistral wind', 'Santa Ana wind', 'autumn', 'Anemometer', 'weather map', 'accumulation', 'chinook wind', 'weather satellite', 'meteorology', 'dew', 'EF-scale', 'eye wall', 'National Hurricane Center (NHC)', 'weather balloon', 'barometric pressure', 'climatology', 'rain shadow', 'newscast ', 'fog bank', 'hydrometer', 'tropical wave', 'Beaufort wind scale', 'weathervane', 'high', 'gale', 'stratosphere', 'warning', 'sun pillar', 'muggy', 'downburst', 'subtropical', 'El Ni\xa4o', 'surge', 'downdraft', 'haboob', 'rainbow', 'Tropic of Capricorn', 'storm tracks', 'rope tornado', 'dust devil', 'winter','radiation', 'snow', 'noreaster', 'flash flood', 'flood', 'breeze', 'cold wave', 'black ice', 'thermometer', 'shower', 'polar front', 'unstable', 'snowflake', 'rainbands', 'blustery', 'polar', 'pileus cloud', 'ice crystals', 'trace', 'normal', 'tornado', 'rain gauge', 'nowcast', 'stationary front', 'drift', 'thunderstorm', 'global warming', 'cloudburst', 'weathering', 'flurry', 'tropical disturbance', 'cyclone ', 'drought', 'wave', 'ice pellets', 'cyclonic flow', 'temperate', 'condensation', 'air pollution', 'relative humidity', 'outflow', 'overcast', 'hydrologic cycle', 'shower sky', 'hurricane season', 'drifting snow', 'warm', 'sunset', 'triple point', 'almanac', 'cold front', 'vapor', 'snow flurry', 'wind']

#count_vectorizer( min_df=1,stop_words =stopwords)

#complete_tweet=train_tweet+test_tweet
#count_vectorizer.fit_transform(complete_tweet)

#train_matrix=count_vectorizer.transform(train_tweet)
#test_matrix=count_vectorizer.transform(test_tweet)

tfidf=TfidfVectorizer(max_features=500, strip_accents='unicode',analyzer='word',stop_words=stopwords)
for_features=columns['tweet'][:n]
tfidf.fit(for_features)
X_tf=tfidf.transform(columns['tweet'][:n])
test_tf=tfidf.transform(columns1['tweet'])
print tfidf.get_feature_names()
#tfidf = TfidfTransformer(norm="l2")
#tfidf.fit(train_matrix)
#train_tfidf=tfidf.idf_

#tfidf.fit(test_matrix)
#test_tfidf=tfidf.idf_

#count_vectorizer.fit_transform(complete_feature)
#complete_feature_matrix=count_vectorizer.transform(complete_feature)

list_index=['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
#list_index=['s3']
#list_index=['ss1','ss2','ss3','ss4','ss5','ww1','ww2','ww3','ww4','kk1','kk2','kk3','kk4','kk5','kk6','kk7','kk8','kk9','kk10','kk11','kk12','kk13','kk14','kk15']
#X=train_matrix

linear=LinearRegression ()
forest=RandomForestRegressor(n_estimators=50, max_features='sqrt',min_samples_split=10)
#boost = GradientBoostingRegressor(n_estimators=500, learning_rate=1.0,max_depth=1, random_state=0, loss='ls')
#X_test=test_matrix # used for prediction
abc=[0]*len(list_index)
X_rf=X_tf.toarray()
test_rf=test_tf.toarray()
for i in range(len(list_index)):
   list=columns[list_index[i]]
   
   list1=[float(j) for j in list]
   
   Y=np.array(list1[:n])
   
   #linear.fit(X,Y)
   #linear.fit(X_tf,Y)
   forest.fit(X_rf,Y)
    #boost.fit(X_rf,Y)
   #pred=boost.predict(test_rf)
   #pred=linear.predict(X_test)
   #pred=linear.predict(test_tf)
   pred=forest.predict(test_rf)
   abc[i]=pred
#print abc[8][8]
print len(abc)
#print abc[4]

#coeff=logistic.coef_.ravel()

#print abc, len(abc)
#print abc[53]


for i in range(len(abc)):
   abc[i]=map(lambda j:float(round(j,3)),abc[i])

#print a
rows=zip(abc[0],abc[0],abc[1],abc[2],abc[3],abc[4],abc[5],abc[6],abc[7],abc[8],abc[9],abc[10],abc[11],abc[12],abc[13],abc[14],abc[15],abc[16],abc[17],abc[18],abc[19],abc[20],abc[21],abc[22],abc[23],abc[23])

with open("D:\twdata\data_test19.txt","w") as file:
    for item in rows:
        print >> file, item

mse=0
#print abc[1]
#print abc[1][1]
#list3=columns1['list_index[j]']
#print list3[1]

#for j in range(24):
#   list3=columns1[list_index[j]]
#   list3=[float(j) for j in list3]
#   #print list3
#   print abc[0]
#   for i in range(n):
         
#      print list3[i]
#      print abc[j][i]
#      mse=mse+(abc[j][i])^2-(list3[i])^2
#rmse=math.sqrt(mse)
#print rmse
              
#with open("\data2.txt","w") as file:
#   for i in range(len(abc)):
#      for item in abc[i]:
#         print >> file, item

print time.clock()-start_time, "seconds"
