import requests, sys
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

def get_champ_list():
  link = 'https://www.op.gg/champion/statistics'
  f = requests.get(link)
  soup = BeautifulSoup(f.text,'lxml')
  content = soup.find('div', 'tabItem ChampionTrendSummary-WinRatio-'+lane.upper())
  return [ champ.text.lower().strip(' ') for champ in content.find_all('div','ChampionName')]

def get_rate(champ_name):
  link = "https://www.op.gg/champion/"+champ_name+"/statistics/"+ lane +"/matchup"
  f = requests.get(link)
  soup = BeautifulSoup(f.text,'lxml')
  content = soup.find('div', {'class': 'SideContent'}).tbody
  v = {}
  for champ in content.find_all('tr'):
    name = champ['data-champion-name']
    rate = champ.find('td','Cell WinRatio').string.strip().strip('%')
    v[name] = float(rate)/100
  v[champ_name] = 0.5
  return v

def get_array():
  A = []
  for champ in champs:
    print champ
    #string = champ
    v = get_rate(champ)
    tmp = []
    for c in champs:
      if c in v:
        tmp.append(v[c])
        #string += ', ' + str(v[c])
      else:
        tmp.append(0.5)
        #string += ', none'
    A.append(tmp)
    #print string
  return np.asarray(A)

def run_HAC(linkage):
  n_topics = 5
  cluster = AgglomerativeClustering(linkage=linkage, n_clusters=n_topics, ).fit(data)
  #for i in xrange(n_topics):
  #  print 'cluster ' + str(i) + ': '+ ' '.join([ champs[j] for j in xrange(len(champs)) if cluster.labels_[j]==i ])
  #print cluster.children_
  return cluster

def generate_newick(cluster):
  string = ''
  d = {}
  n_champ = len(champs)
  for i,name in enumerate(champs):
    d[i]= name
  for i, e in enumerate(cluster.children_):
    d[n_champ+i] = '(' + d[e[0]] + ','+ d[e[1]] +')' + str(n_champ+i)
  return d[2*(n_champ-1)]+';'

  #champs = get_rate('thresh').keys()
lane = sys.argv[1] #'adc' # sys.argv.[1]  adc, support ,middle,top, jungle
champs = get_champ_list()
if lane == 'top':
  champs.remove('warwick')
  champs.remove('hecarim')
print 'Total ' + str(len(champs)) + ' champions in ' + lane +' lane'
print 'Start crawling data from opgg..'
get_array().dump('hero.data') #
data = np.load('hero.data')
print 'Building tree..'
tree = run_HAC('complete')
print 'tree structure has generated. Please copy-paste the belowing sytax to http://etetoolkit.org/treeview/'
print generate_newick(tree)

#X_pca = PCA().fit_transform(data)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=champs)
#plt.show()

#kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(data)
#lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
#                                learning_method='online',
#                                learning_offset=50.,
#                                random_state=0).fit(data)
#nmf = NMF(n_components=n_topics, random_state=1,
#          alpha=.1, l1_ratio=.5).fit(data)

#def print_top_words(model, feature_names, n_top_words):
#  print '[LDA topic model]'
#  for topic_idx, topic in enumerate(model.components_):
#    print("Topic #%d:" % topic_idx)
#    print(" ".join([feature_names[i]
#                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
#  print '\n'

#print_top_words(lda, champs, n_top_words=5)
#print '[K-means clustering]'
