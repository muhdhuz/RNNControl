
import numpy as np
import librosa # for audio file loading and saving

import glob
import os

# if you want 
#defaults give you midi pitch number-to-freq conversions
class octaves () :
		
		def __init__(self, p1=69., p2=81., f1=None,	 f2=None) :
			self.p1 = p1
			self.f1=f1 or 440*np.power(2,(p1-69.)/12) # f1 or the freq for midi notenum p1
			self.p2=p2
			self.f2=f2 or 440*np.power(2,(p2-69.)/12) # f2 or the freq for midi notenum p2
			
			self.prange=self.p2-self.p1
			self.numoctaves=np.log2(self.f2/self.f1)
			self.stepsperoctave=(p2-p1)/self.numoctaves
		#-------------------------------------
		def param2freq(self,p) :
			return self.f1 * np.power(2, (p-self.p1)/self.stepsperoctave)
		pitch2freq=param2freq


		def freq2param (self, f) :
			return np.log2(f/self.f1)*self.stepsperoctave + self.p1
		freq2pitch = freq2param

		#-------------------------------------
		# norm is in [0,1] for p in [p1, p2]
		def param2norm(self, p) :
			return (p-self.p1)/self.prange
		pitch2norm=param2norm

		def norm2param(self, nf) :
			#if nf >=1 :
			#	 nf=.9999999999
			return self.p1+nf*self.prange
		norm2pitch =  norm2param
			
		#-------------------------------------			  
		#returns freq on the octave scale
		def norm2freq(self, nf) :
			#if nf >=1 :
			#	 nf=.9999999
			return self.pitch2freq(self.norm2param(nf))

		#returns the norm on the pitch sclale
		def freq2norm(self, f) :
			return self.param2norm(self.freq2param(f))


#========================================================================================
class linear () :
		
		def __init__(self, p1=0., f1=440., p2=100., f2=880.) :
			self.p1=p1
			self.f1=f1
			self.p2=p2
			self.f2=f2
			
			self.prange=self.p2-self.p1
			self.frange=self.f2-self.f1
		#-------------------------------------
		def param2freq(self,p) :
			return self.f1 + self.frange*(p-self.p1)/self.prange

		def freq2param (self, f) :
			return self.p1 + self.prange*(f-self.f1)/self.frange

		#-------------------------------------			  
		def param2norm(self, p) :
			return (p-self.p1)/self.prange

		def norm2param(self, nv) :
			#if nv >=1 :
			#	 nv=.9999999999
			return self.p1+nv*self.prange
		   
		#-------------------------------------
		#returns freq 
		def norm2freq(self, nv) :
			#if nv >=1 :
			#	 nv=.9999999999
			return self.f1+nv*self.frange

		def freq2norm(self, f) :
			return (f-self.f1)/self.frange

#======================================================================

def strcount(i, numdigits) :
	'''Returns string representation of i, front-padded with 0s to have a length of at least numdigits'''
	return "0"*(numdigits-len(str(i)))+str(i)


class FileSource() :
	def __init__(self, name, datadir, minNum, maxNum, pmapper, firstName="", lastName="", sr=22050, paddedNumLength=3, skipFirstNSamples=0, skipLastNSamples=0) :
		self.name=name	#for personal use: set and queried by users, not the class code
		self.datadir=datadir
		self.minNum=minNum # lowest numbered file to use
		self.source=[None]*(maxNum-minNum+1) #inclusive
		self.firstName=firstName
		self.lastName=lastName
		self.pmapper=pmapper
		self.paddedNumLength=paddedNumLength
		self.skipFirstNSamples=skipFirstNSamples
		self.skipLastNSamples=skipLastNSamples
		self.sr=sr
		self.loadData()

	def num2aidx(self, num) :
		return num-self.minNum
	
	def aidx2num(self, idx) :
		return idx+self.minNum
	

		
	def loadData(self) :
		# Load sample data for traininglibrosa.core.load(params['data_path'] + "/" + fname) 
		for i in range(len(self.source)) :
			fname=self.firstName + strcount(self.aidx2num(i),self.paddedNumLength) + self.lastName
			#print("source " + str(i) + " comes from filename " + fname)
			self.source[i],_=librosa.core.load(self.datadir + "/" + fname, sr=self.sr)
			self.source[i]=self.source[i][self.skipFirstNSamples: len(self.source[i])-self.skipLastNSamples]
			
	def getItem(self, cps, slen, initphase=None) : 
		'''grab a training vector from a random point in a rnadom file, or from initphase in [0,1], where 1 maps to filelength-seqlength'''

		# insist that the cps is very close to one stored in a file
		assert np.abs(self.pmapper.freq2param(cps) - round(self.pmapper.freq2param(cps))) < .01 , \
			"cps = " + str(cps) + ", you requested item "+ str(self.pmapper.freq2param(cps)) + ", but nearest item is " + str(round(self.pmapper.freq2param(cps)))
		notenum=int(round(self.pmapper.freq2param(cps)))

		#remember, randint *excludes* high end val!
		if (initphase != None) :
			samplestartInt = int(initphase*(len(self.source[self.num2aidx(notenum)])-slen))
		else:
			samplestartInt=np.random.randint(len(self.source[self.num2aidx(notenum)])-slen)
		#print("notenum=" + str(notenum) + ", and samplestartInt=" + str(samplestartInt))
		#print("source num = " + str(self.num2aidx(notenum)))
		#print("get item sample start = " + str(samplestartInt))
		return self.source[self.num2aidx(notenum)][samplestartInt:(samplestartInt+slen)]

	def getSource(self, notenum) :
		'''
		Retrive whole source file for notenum
		'''
		return self.source[self.num2aidx(notenum)]

class SyntheticSource() :
	def __init__(self, name, datadir,  minNum, maxNum,	pmapper, firstName="", lastName="", strNumLength=3, sr=22050) :
		self.name=name	 #for personal use: set and queried by users, not the class code
		self.datadir=datadir  #not used by Syntetic Source
		self.harms=4
		self.minNum=minNum
		self.maxNum=maxNum
		self.firstName=firstName
		self.lastName=lastName
		self.pmapper=pmapper
		self.strNumLength=strNumLength
		self.sr=sr
		self.loadData()
	
	def loadData(self) :
		pass
	
	def getItem(self, cps, slen, initphase=None): 
		'''grab a training vector from a random point in source'''
		initphase = initphase or np.random.uniform(0,1)
		
		cycles=cps*slen/self.sr
		rarray = np.sin((np.linspace(initphase, initphase+cycles, slen, endpoint=False)%1)*2*np.pi)
		if self.harms==4 :
			# harmonics are at antiphase to the fundamental to make an interesting waveform shape
			rarray = .25*rarray + \
				.25*np.sin((np.linspace(2*(initphase-.5), 2*(initphase-.5+cycles), slen, endpoint=False)%1)*2*np.pi) + \
				.25*np.sin((np.linspace(3*(initphase-.5), 3*(initphase-.5+cycles), slen, endpoint=False)%1)*2*np.pi) + \
				 .25*np.sin((np.linspace(4*(initphase-.5), 4*(initphase-.5+cycles), slen, endpoint=False)%1)*2*np.pi)
		return rarray

		def getSource(notenum) :
			#return self.getItem(genvals.param2freq(notenum), 22050)
			return self.getItem(self.pmapper.param2freq(notenum), self.sr)
		
		def setHarms(num) :
			self.harms=num
	
	def save(self, path="./temp") :
		for i in range(self.minNum, self.maxNum+1) : # inclusive
			fname="synth."
			fname= self.firstName + strcount(i,self.strNumLength) + self.lastName

			sig=sin_seq_float(self.pmapper.param2freq(i),  phase=None, slen=self.sr, harms=4)
			librosa.output.write_wav(path + "/" + fname, sig, self.sr)



#======================================================================

# Some utility functions
#*************************************
def time_taken(elapsed):
	"""To format time taken in hh:mm:ss. Use with time.monotic()"""
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	return "%d:%02d:%02d" % (h, m, s)

def mydate():
	"""returns current date and time"""
	from datetime import datetime
	return (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def plot_signal(result,start=None,figsize=(20,1),grid=False,logy=False,start_min_max=[-1.,1.]):
	"""Convenience function to plot signals.
	start = a float or list of floats containing the value(s) where a vertical line will be plotted
	"""
	import matplotlib.pylab as plt
	fig, ax = plt.subplots(figsize=figsize)
	if logy:
		ax.semilogy(np.arange(len(result)), result)
	else:
		ax.plot(np.arange(len(result)), result) #just print one example from the batch
	if start is not None:
		if isinstance(start, (list,)):
			for value in start: 
				ax.vlines(x=value,ymin=start_min_max[0], ymax=start_min_max[1], color='r')
		else:
			ax.vlines(x=start,ymin=start_min_max[0], ymax=start_min_max[1], color='r')
	if grid:
		plt.grid()
	plt.show()

def mostRecent(strPat) :
	files = glob.glob(strPat)
	files.sort(key=os.path.getmtime)
	return files[-1]
	
def listDirectory(directory,fileExtList='.wav',regex=None):										 
	"""returns a list of file info objects in directory that contains extension in the list fileExtList (include the . in your extension string)
	and regex if specified
	fileList - fullpath from working directory to files in directory
	fnameList - basenames in directory (including extension)
	regex - a substring in the filename, if unspecified will list all files in directory"""	
	if regex is not None:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')) and (regex in f))] 
	else:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')))] 
	
	fileList = [os.path.join(directory, f) 
			   for f in fnameList
				if os.path.splitext(f)[1] in fileExtList]  
	return fileList , fnameList

def listDirectory_all(directory,topdown=True):
	"""returns a list of all files in directory and all its subdirectories"""
	fileList = []
	fnameList = []
	for root, _, files in os.walk(directory, topdown=topdown):
		for name in files:
			fileList.append(os.path.join(root, name))
			fnameList.append(name)
	return fileList , fnameList

def mass_delete(directory,regex,topdown=True):
	"""deletes all files matching regex in directory and all its subdirectories"""
	deleteList = []
	for root, _, files in os.walk(directory, topdown=topdown):
		for name in files:
			if regex in name:
				deleteList.append(name)
				os.remove(os.path.join(root, name))
	return deleteList

def extract_nsynth_pitch(filename):
	"""function to extract midi pitch from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> 59""" 
	import re
	n = re.findall(r'(?<=-).*?(?=-)', filename)[0]
	if (n[0]=='0') :
		midinum=int(n[1:])
	else :
		midinum=int(n)
	return midinum

def extract_nsynth_instrument(filename):
	"""function to extract midi pitch from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> keyboard_acoustic""" 
	import re
	n = re.search('(.*?_.*?)_', filename).group(1)
	return n

def extract_nsynth_volume(filename):
	"""function to extract volume level from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> 75""" 
	return filename[-6:-4]

def save_obj(obj, dir, name):
	"""to save an obj using pickle"""
	import pickle
	with open(dir + '/' + name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, name):
	"""to load a pickled object"""
	import pickle
	with open(dir + '/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def chunker(seq, size):
	"""returns chunks of length size for given sequence.
	can be used in a loop -
	A = 'ABCDEFG'
	for group in chunker(A, 2):
    	print(group)
	-> 'AB' 'CD' 'EF' 'G'"""
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))
