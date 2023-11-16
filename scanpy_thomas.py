import numpy as np
import scipy.stats as sps
from statsmodels.stats.multitest import multipletests
import sklearn.decomposition as skd
import pandas as pd
from anndata import AnnData
import scanpy as sc
import bbknn
import scvelo as scv
import celltypist
from celltypist import models
from glob import glob
import signal
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(A,color,file,vmax=None,cmap=None,palette=None):
	plt.clf()
	sc.pl.umap(A,color=color,vmax=vmax,cmap=cmap,palette=palette)
	plt.savefig('scanpilot/'+file+'_umap.pdf',format='pdf',bbox_inches='tight')

def plot_umap_labelson(A,file):
	plt.clf()
	sc.pl.umap(A,color='leiden', legend_loc='on data',legend_fontsize=5, title='')
	plt.savefig('scanpilot/'+file+'_umaplabelson.pdf',format='pdf',bbox_inches='tight')

sc.settings.verbosity = 3
def parallelize(func,tasks):
	nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
	def init_worker():
		signal.signal(signal.SIGINT, signal.SIG_IGN)
	p = Pool(nproc, init_worker)
	try:
		return p.map(func,tasks)
	except (KeyboardInterrupt, SystemExit):
		p.terminate()
		p.join()
		sys.exit(1)
	else:
		p.close()
		p.join()


samplenames=["MSG","MSG2","MSG3","MSG4","MSG5","MSG6","MSG7","MSG8","MSG9","MSG10","MSG11","MSG12","MSG13","MSG14","MSG16","MSG17","MSG18","MSG19","MSG20","MSG21","MSG22","MSG23","MSG24","MSG25","MSG26","MSG29","MSG30","MSG31","PBMC","PBMC2","PBMC3","PBMC4","PBMC5","PBMC6","PBMC7","PBMC8","PBMC9","PBMC17","PBMC18","PBMC19","PBMC20","PBMC21","PBMC23","SP1","SP2","SP3","SP4","3717","3738","3735","3750","3751","3740","3743"]
conditions=["PSS","nonPSS","PSS","PSS","PSS","PSS","nonPSS","nonPSS","nonPSS","PSS","nonPSS","PSS","nonPSS","nonPSS","nonPSS","PSS","PSS","nonPSS","PSS","nonPSS","Unknown","nonPSS","HV","HV","HV","nonPSS","PSS","HV","nonPSS","PSS","nonPSS","PSS","PSS","PSS","PSS","nonPSS","nonPSS","nonPSS","PSS","PSS","nonPSS","PSS","nonPSS","nonPSS","PSS","PSS","PSS","HV","PSS","HV","PSS","PSS","PSS","HV"]
tissues=["SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG"]
conditionD = {samplenames[i]:conditions[i] for i in range(len(samplenames))}
tissueD = {samplenames[i]:tissues[i] for i in range(len(samplenames))}
samples = glob('samples/*')
samples = [sample for sample in samples if 'MUC' not in sample]
samples = [sample for sample in samples if '-' not in sample]
samples = [sample for sample in samples if 'pSG' not in sample]
samples.remove('samples/MSG29');samples.remove('samples/MSG22');samples.remove('samples/SP4');samples.remove('samples/MSG6')
samples.remove('samples/MSG');samples.remove('samples/SP1');samples.remove('samples/MSG24');samples.remove('samples/3741')
def loadnorm_data(sample):
	label = sample.split('/')[1]
	adata = sc.read_10x_h5(sample+'/cellbender_filtered.h5')
	adata.var_names_make_unique()
	#adata = sc.read_10x_mtx(sample+'/filtered_feature_bc_matrix/',var_names='gene_symbols',cache=True)
	sc.pp.filter_cells(adata, min_genes=10)
	sc.pp.filter_cells(adata, min_counts=100)
	adata.var['mt'] = adata.var_names.str.startswith('MT-') + adata.var_names.str.startswith('MTRNR')
	adata.var['rp'] = adata.var_names.str.startswith('RPS') + adata.var_names.str.startswith('RPL')
	adata.var['hb'] = adata.var_names.isin(['HBA1','HBA2','HBB'])
	sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','rp', 'hb'], percent_top=None, log1p=False, inplace=True)
	adata = adata[adata.obs.pct_counts_mt < 15, :]
	adata = adata[adata.obs.pct_counts_rp < 50, :]
	adata = adata[adata.obs.pct_counts_hb < 5, :]
	notrp = [gene for gene in adata.var_names if not gene.startswith(('RPS','RPL','MT-','MTRNR','HBA1','HBA2','HBB'))]
	adata = adata[:, notrp]
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	adata.obs['Sample'] = label
	return adata

adata = parallelize(loadnorm_data,samples)
adata = adata[0].concatenate(adata[1:])
adata.obs['Disease'] = adata.obs['Sample'].map(conditionD)
adata.obs['Tissue'] = adata.obs['Sample'].map(tissueD)
metadata = pd.read_csv('METADATA_scRNAseq_MSG.csv',index_col=0)
metadata.index = ['PBMC'] + metadata.index.str.replace('MSG','PBMC')[1:].to_list()
metadata = pd.concat([metadata,pd.read_csv('METADATA_scRNAseq_MSG.csv',index_col=0)])
metadata = metadata[~metadata.index.duplicated()]

def process_adata(A,label,res=1):
	sc.pp.highly_variable_genes(A, min_mean=0.0125, max_mean=3, min_disp=0.5)
	sc.tl.pca(A)
	bbknn.bbknn(A)
	A.uns['neighbors']['params']['metric'] = 'cosine'
	sc.tl.umap(A)
	sc.tl.leiden(A,resolution=res)
	plot_umap(A,['pct_counts_mt','pct_counts_rp','n_genes','total_counts'],'qc_'+label)
	plot_umap(A,'Sample','samples_'+label)
	plot_umap(A,['Tissue','Disease'],'states_'+label)
	plot_umap(A,'leiden','leiden_'+label)
	sc.tl.rank_genes_groups(A,'leiden')
	plt.clf()
	sc.pl.rank_genes_groups(A, n_genes=25, sharey=False)
	plt.savefig('scanpilot/clustergenes_'+label+'.png',format='png',dpi=200,bbox_inches='tight')
	plt.close('all')
	return A

adata = process_adata(adata,'plusnew')
new_cluster_names = ['CD4+\nEEF1+\nT Cells','CD4+\nS100A+\nT Cells','Monocytes','Seromucous Acini','NK Cells','Circulating B Cells','CD8+\nCytotoxic\nT Cells','CD4-CD8-\nT Cells','Macrophages','Fibroblasts','Plasma Cells','Ducts/Ionocytes','Mucous Acini','CD8+\nExhausted\nT Cells','Dendritic\nCells','Endothelium','Pericytes','Myopeithelium','Immune\nProgenitors','Platelets']
adata.rename_categories('leiden', new_cluster_names)
plot_umap_labelson(adata,'allcells')
new_cluster_names = ['CD4+EEF1+ T Cells','CD4+S100A+ T Cells','Myeloid Lineage','Seromucous Acini','NK Cells','Circulating B Cells','CD8+ Cytotoxic T Cells','CD4-CD8- T Cells','Macrophages','Fibroblasts','Plasma Cells','Ducts/Ionocytes','Mucous Acini','CD8+ Exhausted T Cells','Dendritic Cells','Endothelium','Pericytes','Myopeithelium','Immune Progenitor','Platelets']
adata.rename_categories('leiden', new_cluster_names)
adata.write('adata.h5ad')

samples12 = ['MSG5', 'MSG6', 'SP3', 'MSG12', 'MSG10', 'MSG3', 'MSG13', 'MSG7', 'MSG4', 'MSG9', 'MSG8', 'MSG11','3717']
adata12 = parallelize(loadnorm_data,['samples/'+sample for sample in samples12])
adata12 = adata12[0].concatenate(adata12[1:])
adata12.obs['Disease'] = adata12.obs['Sample'].map(conditionD)
adata12.obs['Tissue'] = adata12.obs['Sample'].map(tissueD)
adata12 = process_adata(adata12,'H12')
adata12.write('adataH12.h5ad')

def check_coexpr_and_mt(A,g1,g2,comp,cutoff=1):
	G1 = A.X[:,A.var.index.get_loc(g1)].toarray().T[0]
	G2 = A.X[:,A.var.index.get_loc(g2)].toarray().T[0]
	compDF = A.obs[[comp,'pct_counts_mt']]
	compDF[g1] = G1
	compDF[g2] = G2
	print("MT\% before the cutoff:  "+str(compDF['pct_counts_mt'].mean()))
	compDF = compDF[compDF[g1] > cutoff]
	compDF = compDF[compDF[g2] > cutoff]
	print("MT\% after the cutoff:  "+str(compDF['pct_counts_mt'].mean()))
	print("Coexpression: "+str(np.corrcoef(compDF[g1],compDF[g2])[0][1]))
	compDF = pd.DataFrame(compDF[comp].value_counts())
	compDF['Total'] = A.obs[comp].value_counts()
	print("Proportions of coexpression across "+comp+":")
	print(compDF[comp] / compDF['Total'])

# PULLING THE TWO TISSUE SETS APART

adata = sc.read_h5ad('adata.h5ad')
adata.obs['leiden_before'] = adata.obs['leiden']
adataSG = process_adata(adata[adata.obs['Tissue'] == 'SG'],'SG')
adataPBMC = process_adata(adata[adata.obs['Tissue'] == 'PBMC'],'PBMC')

new_cluster_names = ['Fibroblasts','PRR4+CST3+\nWFDC2-\nSMACs','IgA Plasma Cells','PRR4+CST3+\nWFDC2+\nSMACs','T Cells','PRR4-CST3-\nWFDC2-\nSMACs','Mucous\nAcini','PRR4+CST3-\nWFDC2-\nSMACs','Myoepithelium','Endothelium','Ductal Cells','Pericytes','PRR4-CST3+\nWFDC2- and\nPRR4-CST3-\nWFDC2+\nSMACs','IgG Plasma Cells','','Antigen-Presenting\nCells','Intermediate\nEpithelium','B Cells','High\nZG16B\nSMACs']
adataSG.rename_categories('leiden', new_cluster_names)
plot_umap_labelson(adataSG,'SG')
new_cluster_names = ['Fibroblasts','PRR4+CST3+WFDC2- SMACs','IgA Plasma Cells','PRR4+CST3+WFDC2+ SMACs','T Cells','PRR4-CST3-WFDC2- SMACs','Mucous Acini','PRR4+CST3-WFDC2- SMACs','Myoepithelium','Endothelium','Ductal Cells','Pericytes','PRR4-CST3+WFDC2- SMACs','IgG Plasma Cells','PRR4-CST3-WFDC2+ SMACs','Antigen-Presenting Cells','Intermediate Epithelium','B Cells','High ZG16B SMACs']
adataSG.rename_categories('leiden', new_cluster_names)

adataPBMC.write('adataPBMC.h5ad')

adata12 = sc.read('adataH12.h5ad')
adataSG = sc.read('adataSG.h5ad')
adata = sc.read_h5ad('adata.h5ad')
adataPBMC = sc.read_h5ad('adataPBMC.h5ad')

plot_umap(adataSG,['NEAT1','PRR4','CST3','WFDC2','MUC7','AQP5','ZG16B','FTH1'],'SG_genes1')
plot_umap(adataSG,['GNB2L1','LYZ','C6orf58','KRT14','KRT4','KRT5','MKI67','SLC5A5'],'SG_genes2')
sc.tl.rank_genes_groups(adataSG,'Annotation')
topgenes = adataSG.uns['rank_genes_groups']['names']
plt.clf()
sc.pl.dotplot(adataSG,var_names=list(set(list(topgenes[0])+list(topgenes[1])+list(topgenes[2]))),groupby='Annotation',standard_scale='var',dendrogram=True,figsize=(10,6))
plt.savefig('scanpilot/clustergenes_dotplot_SG.pdf',format='pdf',dpi=200,bbox_inches='tight')

plot_umap(adataSG,'Disease','ss',palette={'nonPSS':'#1f77b4','PSS':'#ff7f0e'})
plot_umap(adataSG,'SSA','ssa',palette={'Positive':'#e41a1c','Negative':'#4daf4a'})

plt.clf()
epi_order = ['Ductal Progenitors','Ionocytes','Ductal Cells','PRR4+CST3+WFDC2+ SMACs','PRR4- Transitioning SMACs','PRR4+CST3+WFDC2- SMACs','PRR4+CST3-WFDC2- SMACs','PRR4-CST3-WFDC2- SMACs','High ZG16B SMACs','Transitioning Mucous Acini','Terminal Mucous Acini','Intermediate Epithelium']
adataEpi = adataSG[adataSG.obs['Annotation'].isin(epi_order)]
sc.pl.stacked_violin(adataEpi, ['WFDC2','CST3','AQP5','PRR4','ZG16B','LYZ','MUC5B','MUC7'], groupby='Annotation',categories_order=epi_order)
plt.savefig('scanpilot/umap_SG_sv1.png',format='png',dpi=300,bbox_inches='tight')
plt.close('all')

adata12.obs['leiden'] = adata12.obs['leiden'].astype('int')
new_cluster_names = ['T Cells','Fibroblasts','Plasma Cells','Mucous Acini','Seromucous Acini','Ducts','Seromucous Acini','Seromucous Acini','Seromucous Acini','Endothelium','Seromucous Acini','Plasma Cells','Plasma Cells','B Cells','Antigen-Presenting Cells','Seromucous Acini','Pericytes','Seromucous Acini','Seromucous Acini','Myopeithelium','Mucous Acini','Seromucous Acini','Ionocytes','Seromucous Acini']
for i in range(len(adata12.obs['leiden'].value_counts())):
	adata12.obs['leiden'][adata12.obs['leiden'] == i] = new_cluster_names[i]

adata12.obs['leiden'] = adata12.obs['leiden'].astype('category')
plot_umap_labelson(adata12,'H12')

# SLICING UMAPS

X_umap = adataSG[adataSG.obs['leiden'] == 'T Cells'].obsm['X_umap']
colors = []
for x in X_umap:
	if 2.75*x[0]+x[1] > 19.5:
		colors.append('#ff0000')
	elif x[1] - 2.5*x[0] > - 25.3:
		colors.append('#00ff00')
	else:
		colors.append('#0000ff')

plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.1,c=colors)
plt.savefig('test_umap.png',format='png',dpi=200,bbox_inches='tight')

adataAPC = adataSG[adataSG.obs['leiden'] == 'Antigen-Presenting Cells']
X_umap = adataAPC.obsm['X_umap']
colors = []; labels = []
for x in X_umap:
	if x[0] < 15.6:
		colors.append('#ff0000')
		labels.append('Finger 1')
	elif x[1]+7.2*x[0] > 117.7:
		colors.append('#00ff00')
		labels.append('Finger 3')
	else:
		colors.append('#0000ff')
		labels.append('Finger 2')

plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.1,c=colors)
plt.savefig('test_umap2.png',format='png',dpi=200,bbox_inches='tight')

adataAPC.obs['finger_labels'] = labels
sc.tl.rank_genes_groups(adataAPC,'finger_labels')
plt.clf()
sc.pl.rank_genes_groups(adataAPC)
plt.savefig('test_umap3.png',format='png',dpi=200,bbox_inches='tight')

X_umap = adataSG[adataSG.obs['leiden'] == 'Ductal Cells'].obsm['X_umap']
colors = []
for x in X_umap:
	if x[0] > 11 and x[1] > 15.3:
		colors.append('#ff0000')
	elif x[0] > 11.2 and x[1] < 14.2:
		colors.append('#00ff00')
	else:
		colors.append('#0000ff')

plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.1,c=colors)
plt.savefig('test_umap4.png',format='png',dpi=200,bbox_inches='tight')

X_umap = adataSG[adataSG.obs['leiden'] == 'Mucous Acini'].obsm['X_umap']
colors = ['#dd0000' if x[1] > 16.9 else '#00dd00' for x in X_umap]
plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.05,c=colors)
plt.savefig('test_umap5.png',format='png',dpi=200,bbox_inches='tight')

X_umap = adataSG[adataSG.obs['leiden'] == 'Pericytes'].obsm['X_umap']
colors = ['#dd0000' if x[0] > 21.6 else '#00dd00' for x in X_umap]
plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.05,c=colors)
plt.savefig('test_umap6.png',format='png',dpi=200,bbox_inches='tight')

X_umap = adataSG[adataSG.obs['leiden'] == 'Endothelium'].obsm['X_umap']
colors = []
for x in X_umap:
	if x[1]-(4/3)*x[0] < -22.7:
		colors.append('#ff0000')
	elif x[0] < 20.9:
		colors.append('#00ff00')
	else:
		colors.append('#0000ff')

plt.clf()
plt.scatter(X_umap.T[0],X_umap.T[1],s=1,alpha=0.05,c=colors)
plt.savefig('test_umap7.png',format='png',dpi=200,bbox_inches='tight')

muc7 = adataSG.X[:,adataSG.var.index.get_loc('MUC7')].toarray().T[0]

Annotation = list(adataSG.obs['leiden'])
gnly = adataSG.X[:,adataSG.var.index.get_loc('GNLY')].toarray().T[0]
foxp3 = adataSG.X[:,adataSG.var.index.get_loc('FOXP3')].toarray().T[0]
cd24 = adataSG.X[:,adataSG.var.index.get_loc('CD24')].toarray().T[0]
for i in range(adataSG.obs.shape[0]):
	if adataSG.obs['leiden'][i] == 'T Cells':
		if foxp3[i] > 1:
			Annotation[i] = 'Regulatory T Cells'
		elif cd24[i] > 1:
			Annotation[i] = 'T Cell Progenitors'
		elif gnly[i] > 4:
			Annotation[i] = 'NK Cells'
		elif 2.75*adataSG.obsm['X_umap'][i][0]+adataSG.obsm['X_umap'][i][1] > 19.5:
			Annotation[i] = 'CD8+ Exhausted T Cells'
		elif adataSG.obsm['X_umap'][i][1] - 2.5*adataSG.obsm['X_umap'][i][0] > - 25.3:
			Annotation[i] = 'CD4+ T Cells'
		else:
			Annotation[i] = 'CD8+ Effector T Cells'
	elif adataSG.obs['leiden'][i] == 'Ductal Cells':
		if adataSG.obsm['X_umap'][i][0] > 11 and adataSG.obsm['X_umap'][i][1] > 15.3:
			Annotation[i] = 'Ionocytes'
		elif adataSG.obsm['X_umap'][i][0] > 11.2 and adataSG.obsm['X_umap'][i][1] < 14.2:
			Annotation[i] = 'Ductal Progenitors'
	elif adataSG.obs['leiden'][i] == 'Antigen-Presenting Cells':
		if adataSG.obsm['X_umap'][i][0] < 15.6:
			Annotation[i] = 'M1 Macrophages'
		elif adataSG.obsm['X_umap'][i][1]+7.2*adataSG.obsm['X_umap'][i][0] > 117.7:
			Annotation[i] = 'M2 Macrophages'
		else:
			Annotation[i] = 'Dendritic Cells'
	elif adataSG.obs['leiden'][i] == 'Mucous Acini':
		if adataSG.obsm['X_umap'][i][1] > 16.9:
			Annotation[i] = 'Terminal Mucous Acini'
		else:
			Annotation[i] = 'Transitioning Mucous Acini'
	elif adataSG.obs['leiden'][i] == 'Pericytes':
		if adataSG.obsm['X_umap'][i][0] < 21.6:
			Annotation[i] = 'Smooth Muscle'
	elif adataSG.obs['leiden'][i] == 'Endothelium':
		if adataSG.obsm['X_umap'][i][1]-(4/3)*adataSG.obsm['X_umap'][i][0] < -22.7:
			Annotation[i] = 'Venules'
		elif adataSG.obsm['X_umap'][i][0] < 20.9:
			Annotation[i] = 'Arterioles'
		else:
			Annotation[i] = 'Capillaries'
	elif adataSG.obs['leiden'][i] == 'PRR4-CST3+WFDC2- SMACs':
		Annotation[i] = 'PRR4- Transitioning SMACs'
	elif adataSG.obs['leiden'][i] == 'PRR4-CST3-WFDC2+ SMACs':
		Annotation[i] = 'PRR4- Transitioning SMACs'

Annotation = pd.Series(Annotation).astype('category')
Annotation.index = adataSG.obs.index
adataSG.obs['Annotation'] = Annotation
plot_umap(adataSG,'Annotation','Annotation')
adataSG.write('adataSG.h5ad')

# MAKING COMPARISONS

plot_umap(adataSG,['CLDN5','AQP1','CA4','ITGA8','MCAM','GFRA3','ADIRF','MKI67'],'sgonly_genes1')
plot_umap(adataSG,['GNLY','GZMA','GZMB','GZMK','CD8A','CD3G','CD4'],'sgonly_genes2')
plot_umap(adataSG,['EOMES','PDCD1','ITGA1','SPRY1','CD40LG','CD24','IL2RA','FOXP3'],'sgonly_genes3')
plot_umap(adataSG,['KRT15','SOX2','FOXI1','CFTR','ASCL3','DCN','LUM'],'sgonly_genes4')
plot_umap(adataSG,['MUC7','ODAM','MUC5B','BPIFB2','S100A2','WFDC2','KRT14','ACTA2'],'sgonly_genes5')
plot_umap(adataSG,['CHGA','GFRA3','CD1C','CD163','FCGR3A','IFNGR1','CX3CR1','MS4A1'],'sgonly_genes6')
plot_umap(adataSG,['IGKC','BMP6'],'sgonly_genes7')
plot_umap(adataSG,['IGKC','IGLC2','IGHA1','IGHM','IGHD','IGHG1'],'sgonly_genes8')
plot_umap(adataSG,['NPR3','BOC','RAPGEF5','DSG1','LYPD3','FAM46B','PTHLH','SLC1A3'],'sgonly_genes9')
plot_umap(adataSG,['FGFR2','CCNB1','HMGB3','CENPW','SPRR1B','KRT76','KRT6B','WFDC2'],'sgonly_genes10')
plot_umap(adataSG,['LCN2','SCGB3A1','CEACAM6','NKX3-1','TSPAN8','KRT17','EDN3','FBXO32'],'sgonly_genes11')
plot_umap(adataSG,['ATP6V1B1','GPRC6A','MECOM','SEMA3G','TSPAN2','RBP7','TMEM88','ACKR1'],'sgonly_genes12')
plot_umap(adataSG,['SELP','IL33','CCL21','MMRN1','PROX1','DPT','CFH','STEAP4'],'sgonly_genes13')
plot_umap(adataSG,['ABCC9','MYO1B','RERGL','PLN','DCT','MLANA','TYR','CHGA'],'sgonly_genes14')
plot_umap(adataSG,['CCER2','VIP','CD8B','BATF','CTLA4','TRGC2','TRG-AS1'],'sgonly_genes15')
plot_umap(adataSG,['KLRC1','XCL2','FAM111B','CLSPN','ATAD5','BANK1','TNFRSF13C','TNFRSF17'],'sgonly_genes16')
plot_umap(adataSG,['AL928768.3','DERL3','LILRA4','SCT','PLAC8','XCR1','C1orf54','WDFY4'],'sgonly_genes17')
plot_umap(adataSG,['CLEC10A','FCN1','CD1C','CD207','CD1A','PKIB','FOLR2','CD163'],'sgonly_genes18')
plot_umap(adataSG,['F13A1','FCGR3A','CCL3','CCL3L3','TPSB2','CPA3','TPSAB1'],'sgonly_genes19')
plot_umap(adataSG,['SEMA3G','GJA4','GJA5','CXCL12','HEY1','NOTCH4','ASS1','S100A4'],'sgonly_genes20')
plot_umap(adataSG,['RGCC','EFNB2','SOX17','PLAUR','KLHL21','GJA1','LITAF','ICAM1'],'sgonly_genes21')
plot_umap(adataSG,['IRF1','SELE','SELP','LRG1','EGR2','ACKR1','CCL14','CLU'],'sgonly_genes22')
plot_umap(adataSG,['ACTA2','CCL19','CCL21','PTH1R','FIBIN','PTGFR','MAB21L1'],'sgonly_genes23')
plt.close('all')

def compare(adata,group1,group2):
	plt.clf()
	sc.tl.rank_genes_groups(adata, 'Annotation', groups=[group2], reference=group1)
	res1 = list(adata.uns['rank_genes_groups']['names'][:10])
	sc.pl.rank_genes_groups(adata, groups=[group2], n_genes=20)
	plt.savefig('scanpilot/compare/'+group2+'_'+group1+'.pdf',format='pdf',dpi=200,bbox_inches='tight')
	plt.clf()
	sc.tl.rank_genes_groups(adata, 'Annotation', groups=[group1], reference=group2)
	res2 = list(adata.uns['rank_genes_groups']['names'][:10])
	sc.pl.rank_genes_groups(adata, groups=[group1], n_genes=20)
	plt.savefig('scanpilot/compare/'+group1+'_'+group2+'.pdf',format='pdf',dpi=200,bbox_inches='tight')
	return res1,res2

celltypes = adataSG.obs['Annotation'].unique().astype('str')
celltypeD = {celltype:[] for celltype in celltypes}
for i in range(len(celltypes)):
	for j in range(i):
		r1, r2 = compare(adataSG,celltypes[i],celltypes[j])
		celltypeD[celltypes[i]] += r2
		celltypeD[celltypes[j]] += r1

for celltype in celltypeD:
	top5 = pd.Series(celltypeD[celltype]).value_counts().index
	print('\t'.join(top5)+'\t'+celltype)

for i in celltypes:
	tempadata = adataSG[adataSG.obs['Annotation'] == i]
	tempadata.obs['Disease'][tempadata.obs['Disease'] == 'HV'] = 'nonPSS'
	sc.tl.rank_genes_groups(tempadata,'Disease',groups=['PSS'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/compare/genes_SG_'+i+'_1.pdf',format='pdf',dpi=200,bbox_inches='tight')
	sc.tl.rank_genes_groups(tempadata,'Disease',groups=['nonPSS'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/compare/genes_SG_'+i+'_2.pdf',format='pdf',dpi=200,bbox_inches='tight')

for i in celltypes:
	tempadata = adataSG[adataSG.obs['Annotation'] == i]
	tempadata = tempadata[tempadata.obs['Sample'].map(metadata['Sex'].to_dict()) == 0] #necessary otherwise XIST is a major difference
	sc.tl.rank_genes_groups(tempadata,'SSA',groups=['Positive'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/compare/ssa_SG_'+i+'_1.pdf',format='pdf',dpi=200,bbox_inches='tight')
	sc.tl.rank_genes_groups(tempadata,'SSA',groups=['Negative'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/compare/ssa_SG_'+i+'_2.pdf',format='pdf',dpi=200,bbox_inches='tight')

DF = adataSG.obs
b2m = adataSG.X[:,adataSG.var.index.get_loc('B2M')].toarray().T[0]
DF['Disease'][DF['Disease'] == 'HV'] = 'nonPSS'
DF['Disease'].cat.remove_unused_categories()
DF['B2M'] = b2m
plt.clf()
ax = sns.violinplot(data=DF,y='B2M',x='leiden',hue='Disease',split=True,hue_order=['nonPSS','PSS'])
ax.set_xticklabels(DF['leiden'].value_counts().index, rotation = 45, ha="right")
plt.savefig('scanpilot/umap_SG_B2Mdisease.png',format='png',dpi=200,bbox_inches='tight')

#change barplots to deltas - somehow
adataSG.obs['SSA'] = adataSG.obs['Sample'].map((metadata['SSA'] > 0).to_dict())

# What are the differences between actively secreting cells and others?
tempadata = adataSG[adataSG.obs['leiden'].isin(['PRR4+CST3-WFDC2- SMACs','PRR4-CST3-WFDC2- SMACs','IgA Plasma Cells','T Cells'])]
tempadata.obs['is T'] = (tempadata.obs['leiden'] == 'T Cells').astype('category')
sc.tl.rank_genes_groups(tempadata,'is T')
plt.clf()
sc.pl.rank_genes_groups(tempadata)
plt.savefig('scanpilot/whysecretionisspecial.pdf',format='pdf',dpi=200,bbox_inches='tight')
plot_umap(adataSG,'SSR4','whysecretionisspecial')

tempadata = adataSG[adataSG.obs['leiden'] == 'PRR4+CST3+WFDC2- SMACs']
tempadata.obs['AQP5+'] = pd.Series(tempadata.X[:,tempadata.var.index.get_loc('AQP5')].toarray().T[0] > 2,index = tempadata.obs.index).astype('category')
#tempadata.obs['CST3+'] = (tempadata.obs['leiden'].str.contains('CST3+')).astype('category')
sc.tl.rank_genes_groups(tempadata,'AQP5+')
plt.clf()
sc.pl.rank_genes_groups(tempadata)
plt.savefig('scanpilot/whyaqp5isspecial.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(4,6))
a = adataSG[adataSG.obs['Annotation'].isin(['NK Cells','B Cells'])]
aP = a[a.obs['SSA'] == 'Positive']
aN = a[a.obs['SSA'] == 'Negative']
ssagenes = ['STATH','AC090498.1','BST2','CD81','S100A10','KIAA0226L','GYPC','PLEK','MYOM2']
ax1 = sc.pl.dotplot(aP, ssagenes, groupby='Annotation', ax=ax1,show=False)
ax2 = sc.pl.dotplot(aN, ssagenes, groupby='Annotation', ax=ax2,show=False)
plt.savefig('scanpilot/ssa_dotplot.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,6))
a = adataSG[adataSG.obs['Annotation'].isin(['Ductal Progenitors','Ionocytes','Pericytes','PRR4+CST3+WFDC2+ SMACs','PRR4+CST3+WFDC2- SMACs','Transitioning Mucous','Terminal Mucous'])]
aP = a[a.obs['Disease'] == 'PSS']
aN = a[a.obs['Disease'] == 'nonPSS']
ssgenes = ['B2M','TPT1','SOX9','JUND','UBA52','CLDN3','HSP90AB1','NCL','ZFP36L1','GNAS','SSR4','CLDN10','TMED2','BPIFB2']
ax1 = sc.pl.dotplot(aP, ssgenes, groupby='Annotation', ax=ax1,show=False)
ax2 = sc.pl.dotplot(aN, ssgenes, groupby='Annotation', ax=ax2,show=False)
plt.savefig('scanpilot/ss_dotplot.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,8))
a = adataSG[adataSG.obs['Annotation'].isin(['High ZG16B SMACs','Intermediate Epithelium','Dendritic Cells','M1 Macrophages','M2 Macrophages','Regulatory T Cells'])]
aP = a[a.obs['Disease'] == 'PSS']
aN = a[a.obs['Disease'] == 'nonPSS']
aPP = aP[aP.obs['SSA'] == 'Positive']
aPN = aP[aP.obs['SSA'] == 'Negative']
aNP = aN[aN.obs['SSA'] == 'Positive']
aNN = aN[aN.obs['SSA'] == 'Negative']
bothgenes = ['PLCG2','SCGB3A1','CXCL14','AZGP1','ZG16B','SLPI','PIP','C6orf58','LYZ','NFKBIA','IER3','BTG2','ZFP36','EEF1A1','HLA-DQA2','HLA-DQA1','HLA-B','PSEN2','LY6E','ADAMDEC1','TMEM176A','CNIH1','ANXA1','DBI','TUBB4B','UBL5','SEPT7']
ax1 = sc.pl.dotplot(aPP, bothgenes, groupby='Annotation', ax=ax1,show=False)
ax2 = sc.pl.dotplot(aPN, bothgenes, groupby='Annotation', ax=ax2,show=False)
ax3 = sc.pl.dotplot(aNP, bothgenes, groupby='Annotation', ax=ax3,show=False)
ax4 = sc.pl.dotplot(aNN, bothgenes, groupby='Annotation', ax=ax4,show=False)
plt.savefig('scanpilot/ssass_dotplot.pdf',format='pdf',dpi=200,bbox_inches='tight')


# TEST PROPORTIONS - NEW

def test_proportions(A,test_dict,metadata,label,qonly=True):
	cross = pd.crosstab(A.obs['Annotation'],A.obs['Sample']).T
	cross = cross.divide(cross.sum(axis=1),axis=0)
	cross.columns = list(cross.columns)
	cross['Condition'] = cross.index.map(test_dict)
	if len(cross['Condition'].value_counts()) > 1:
		condition1 = cross['Condition'].value_counts().index[0]
		condition2 = cross['Condition'].value_counts().index[1]
		cross_for_corr = pd.concat([cross,metadata],axis=1).dropna()
		plt.clf()
		sns.clustermap(cross_for_corr.corr(),xticklabels=True,yticklabels=True)
		plt.savefig('scanpilot/cross/'+label+'_corr.pdf',format='pdf',dpi=600,bbox_inches='tight')
		pvals = []
		for col in cross.drop('Condition',axis=1).columns:
			pvals.append(sps.ttest_ind(cross[col][cross['Condition'] == condition1],cross[col][cross['Condition'] == condition2])[1])
		qvals = multipletests(pvals,method='fdr_bh')[1]
		heatmap = cross.groupby('Condition').mean().T
		heatmap['FDR-Corrected Q'] = qvals
		heatmap['P'] = pvals
		if qonly:
			result_columns = list(cross.columns.drop('Condition')[np.array(pvals) < .05])
		else:
			result_columns = list(cross.columns.drop('Condition'))
		if len(result_columns) > 0:
			results = heatmap.loc[result_columns]
			cross = cross[result_columns+['Condition']].melt(id_vars=['Condition'])
			cross.columns = ['Condition','Cell Type','Proportion']
			idx = cross['Condition'].value_counts().index
			plt.clf()
			plt.figure(figsize=(2+(0.3*len(result_columns)),5))
			g = sns.boxplot(data=cross,x='Cell Type',y='Proportion',hue='Condition',hue_order=sorted(idx,key=str.lower),palette=['#1f77b4','#ff7f0e','#d62728'])
			if qonly:
				sns.swarmplot(data=cross,x='Cell Type',y='Proportion',hue='Condition',hue_order=sorted(idx,key=str.lower),palette=['#1f77b4','#ff7f0e','#d62728'],dodge=True)
			handles, labels = g.get_legend_handles_labels()
			l = plt.legend(handles[0:len(idx)], labels[0:len(idx)], bbox_to_anchor=(.8, .99), loc=2, borderaxespad=0.)
			plt.xticks(rotation=45,ha='right')
			plt.savefig('scanpilot/cross/'+label+'_box.pdf',format='pdf',dpi=600,bbox_inches='tight')
			print(results)

metadata = pd.read_csv('METADATA_scRNAseq_MSG.csv',index_col=0)
metadata = metadata[['SSA']].astype('bool').replace(False,'Negative').replace(True,'Positive')
metadata['Disease'] = metadata.index.map(conditionD)
metadata['Disease'] = metadata['Disease'].replace('HV','nonPSS')
condition_dict = metadata['SSA'].to_dict()
metadata['Clinical'] = (metadata['Disease']+metadata['SSA']).replace('nonPSSNegative','nonPSS').replace('nonPSSPositive','nonPSS')
test_proportions(adataSG,metadata['SSA'].to_dict(),metadata,'SSA')
test_proportions(adataSG,metadata['Disease'].to_dict(),metadata,'Disease')
test_proportions(adataSG,metadata['Disease'].to_dict(),metadata,'Disease_all',qonly=False)
test_proportions(adataSG,metadata['Clinical'].to_dict(),metadata,'Clinical')
test_proportions(adataSG,metadata['Clinical'].to_dict(),metadata,'Clinical_all',qonly=False)
plt.close('all')

# LOOKING FOR INDICES

def read_index(name):
	CSV = pd.read_csv(name+'.csv')
	v = []
	for val in CSV['Genes'].dropna().values:
		v += val.split(';')
	return v

indexD = {"IFN Response":['EPSTI1','HERC5','IFI27','IFI44','IFI44L','IFI6','IFIT1','IFIT3','ISG15','LAMP3','LY6E','MX1','OAS1','OAS2','OAS3','PLSCR1','RSAD2','RTP4','SIGLEC1','SPATS2L','USP18']}
indexD["Salivary Secretion KEGG"] = ['AMY1A','CD38','FXYD2','ATP1A','ATP1B','CALM','PRKCA','CHRM3','ADRA1A','ADRA1B','ADRA1D','ADRB1','ADRB2','ADRB3','PKA','GNAS','GNAQ','KCNMA1','KCNN4','ITPR1','ITPR2','ITPR3','RYR3','TRPV6','SLC9A1','ATP2B','PLCB','PRKG1','ADCY1','ADCY2','ADCY3','ADCY4','ADCY5','ADCY6','ADCY7','ADCY8','ADCY9','AQP5','SLC12A2','GUCY1A','GUCY1B','LPO','NOS1','VAMP2','SLC4A2','BEST2','CST1','CST2','CST3','CST4','CST5','MUC5B','MUC7','PRH1','PRB1','DMBT1','HTN','STATH','LYZ','CAMP','BST1','PRKG2','PRKCB','PRKCG']
indexD['Yin et al. 2021'] = ['SETD8','NUMB','BMP3','PRKD1','TGFB2','GSC','FZD7','MT1F','CXCL12','ESR1','CDK4','MST1R','FGFR2','CDH1','CDH11','WNT4']
for filename in ['reactome.gmt','h.all.v7.1.symbols.gmt','c5.bp.v6.2.symbols.gmt.txt','c2.cp.kegg.v6.2.symbols.gmt.txt']:
	with open('secretomes/'+filename) as openfile:
		for line in openfile.read().split('\n')[:-1]:
			line = line.split('\t')
			indexD[line[0]] = line[2:]

for name in ['Fibr Irradiation','Fibr Atazanivir','Fibr RAS Overexpr','Renep Irradiation','Renep RAS Overexpr']:
	indexD[name] = read_index('secretomes/'+name.lower().replace(' ','_')+'_secretome')

with open('secretomes/btm_annotation_table.txt') as openfile:
	for line in openfile.read().split('\n')[1:]:
		line = line.split('\t')
		indexD[line[1]] = line[7].replace('"','').split(',')

concattable = []
for key in indexD:
	concattable.append(pd.DataFrame(True,columns=[key],index=indexD[key]))

indexDF = pd.concat(concattable,axis=1)

def zscore(a):
	a = sps.zscore(a)
	a = np.nan_to_num(a)
	return a

Z = adataSG.X.toarray()
Z = np.apply_along_axis(zscore,0,Z)

indexDF = indexDF.reindex(adataSG.var.index).fillna(False)
pathZ = np.dot(Z,indexDF.values)
DFZ = pd.DataFrame(pathZ,index=adataSG.obs.index,columns=indexDF.columns)
DFZ = DFZ.subtract(DFZ.min())
adataP = AnnData(X=DFZ,obs=adataSG.obs,var=indexDF.astype('int').T[['OR4F5']])
adataP.obs['leiden_SG'] = adataP.obs['leiden']
adataP.obs['disease'] = adataP.obs['Disease']
adataP.obs = adataP.obs.drop('Disease',axis=1)
sc.pp.log1p(adataP)
adataP = process_adata(adataP,'SG_Index')
adataP.write('adataP.h5ad')
adataP = sc.read('adataP.h5ad')
metadata = pd.read_csv('METADATA_scRNAseq_MSG.csv',index_col=0)
adataP.obs['SSA'] = adataP.obs['Sample'].map((metadata['SSA'] > 0).to_dict()).astype('category')
adataP.uns['log1p'] = {'base':None}

plt.clf()
plt.figure(figsize=(7,7))
crosstab = pd.crosstab(adataP.obs['Annotation'],adataP.obs['leiden'])
sns.heatmap(crosstab.divide(crosstab.sum()),cmap='pink')
plt.savefig('scanpilot/Index_crosstab.pdf',format='pdf',dpi=200,bbox_inches='tight')
plot_umap(adataP,'disease','ss_index',palette={'nonPSS':'#1f77b4','PSS':'#ff7f0e'})
plot_umap(adataP,'SSA','ssa_index',palette={'Positive':'#e41a1c','Negative':'#4daf4a'})

ifnDF = pd.DataFrame(np.array([np.e**adataP.X[:,0]-1,adataP.obs['SSA'],adataP.obs['Annotation']]).T,index=adataP.obs.index,columns=['IFN Response','SSA','Annotation'])
plt.clf()
fig=plt.figure(figsize=(10,5))
sns.barplot(data=ifnDF,x='Annotation',y='IFN Response',hue='SSA')
plt.xticks(rotation=45,ha='right')
plt.savefig('scanpilot/ifn_barplot.pdf',format='pdf',bbox_inches='tight')
adataSG.obs['IFN Response'] = ifnDF['IFN Response']
plot_umap(adataSG[adataSG.obs['SSA'] == 'Positive'],'IFN Response','IFN Response Positive')
plot_umap(adataSG[adataSG.obs['SSA'] == 'Negative'],'IFN Response','IFN Response Negative')


def compare(adata,group1,group2):
	plt.clf()
	sc.tl.rank_genes_groups(adata, 'leiden', groups=[group2], reference=group1)
	res1 = list(adata.uns['rank_genes_groups']['names'][:10])
	sc.pl.rank_genes_groups(adata, groups=[group2], n_genes=20)
	plt.savefig('scanpilot/compare/index_'+group2+'_'+group1+'.pdf',format='pdf',dpi=200,bbox_inches='tight')
	plt.clf()
	sc.tl.rank_genes_groups(adata, 'leiden', groups=[group1], reference=group2)
	res2 = list(adata.uns['rank_genes_groups']['names'][:10])
	sc.pl.rank_genes_groups(adata, groups=[group1], n_genes=20)
	plt.savefig('scanpilot/compare/index_'+group1+'_'+group2+'.pdf',format='pdf',dpi=200,bbox_inches='tight')
	return res1,res2

compare(adataP,'3','13')
compare(adataP,'7','9')
eps = ['0','1','2','4','5','6','12']; epD = {ep:[] for ep in eps}
for i in range(len(eps)):
	for j in range(i):
		r1,r2 = compare(adataP,eps[i],eps[j])
		epD[eps[i]] += r2
		epD[eps[j]] += r1

for ep in epD:
	top5 = pd.Series(epD[ep]).value_counts().index[:5]
	print(ep)
	print(top5)

d = pd.crosstab(adataP.obs['disease'],adataP.obs['leiden'])
d = d.loc['SjD']/d.loc['non-SjD']
s = pd.crosstab(adataP.obs['SSA'],adataP.obs['leiden'])
s = s.loc['Positive']/s.loc['Negative']
labels = ['0','1','2','Fibroblasts','4','5','6','Activated T/APCs','Plasma Cells','Inactive T Cells','Contractile Cells','Endothelium','12','13']
plt.clf()
plt.figure(figsize=(5,5))
for i in d.index.tolist():
	plt.scatter(d[i],s[i],color=sc.pl.palettes.default_20[int(i)],label=i)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Ratio SjD/non-SjD');plt.ylabel('Ratio SSA+/SSA-')
plt.savefig('scanpilot/specifity_SG_index.pdf',format='pdf',dpi=200,bbox_inches='tight')
plot_umap(adataP,'SSA','SSA_SG_Index')

adataP.obs['WUS'] = adataP.obs['Sample'].map(metadata['WUS'])
adataP.obs['WSSF'] = adataP.obs['Sample'].map(metadata['WSSF'])

sc.tl.rank_genes_groups(adataP, 'disease')
plt.clf()
sc.pl.rank_genes_groups_stacked_violin(adataP, n_genes=10)
plt.savefig('scanpilot/allindices_DEI_disease.pdf',format='pdf',dpi=200,bbox_inches='tight')
sc.tl.rank_genes_groups(adataP, 'SSA')
plt.clf()
sc.pl.rank_genes_groups_stacked_violin(adataP, n_genes=10)
plt.savefig('scanpilot/allindices_DEI_SSA.pdf',format='pdf',dpi=200,bbox_inches='tight')
adataP.obs['Clinical'] = adataP.obs['disease'].replace('HV','nonPSS').astype('str') + adataP.obs['SSA'].astype('str')
adataP.obs['Clinical'] = adataP.obs['Clinical'].replace('nonPSSNegative','nonPSS').replace('nonPSSPositive','nonPSS')
sc.tl.rank_genes_groups(adataP, 'Clinical')
plt.clf()
sc.pl.rank_genes_groups_stacked_violin(adataP, n_genes=10)
plt.savefig('scanpilot/allindices_DEI_disease+ssa.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
sns.heatmap(pd.crosstab(adataP.obs['leiden_SG'],adataP.obs['leiden']))
plt.savefig('scanpilot/allindices_crosstab.pdf',format='pdf',dpi=200,bbox_inches='tight')
adataT = adataP[adataP.obs['leiden_SG'] == 'T Cells']
adataT = adataT[adataT.obs['leiden'].isin(['7','9'])]
plt.clf()
sns.heatmap(pd.crosstab(adataT.obs['Annotation'],adataT.obs['leiden']))
plt.savefig('scanpilot/allindices_crosstab2.pdf',format='pdf',dpi=200,bbox_inches='tight')

celltypes = adataP.obs['Annotation'].unique().astype('str')
for i in celltypes:
	tempadata = adataP[adataP.obs['Annotation'] == i]
	tempadata.obs['disease'][tempadata.obs['disease'] == 'HV'] = 'nonPSS'
	sc.tl.rank_genes_groups(tempadata,'disease',groups=['PSS'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/index/indices_SG_'+i+'_1.pdf',format='pdf',dpi=200,bbox_inches='tight')
	sc.tl.rank_genes_groups(tempadata,'disease',groups=['nonPSS'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/index/indices_SG_'+i+'_2.pdf',format='pdf',dpi=200,bbox_inches='tight')

tempadata = adataP[adataP.obs['Annotation'] == 'PRR4+CST3+WFDC2- SMACs']
tempadata.obs['disease'][tempadata.obs['disease'] == 'HV'] = 'nonPSS'
sc.tl.rank_genes_groups(tempadata,'disease')
plt.clf()
sc.pl.rank_genes_groups_dotplot(tempadata, n_genes=10,groupby='disease')
plt.savefig('scanpilot/indices_++-.pdf',format='pdf',dpi=200,bbox_inches='tight')
tempadata.obs['Clinical'] = tempadata.obs['disease'].replace('HV','nonPSS').astype('str') + tempadata.obs['SSA'].astype('str')
tempadata.obs['Clinical'] = tempadata.obs['Clinical'].replace('nonPSSNegative','nonPSS').replace('nonPSSPositive','nonPSS')
sc.tl.rank_genes_groups(tempadata,'Clinical')
plt.clf()
sc.pl.rank_genes_groups_dotplot(tempadata, n_genes=10,groupby='Clinical')
plt.savefig('scanpilot/indices_++-ssass.pdf',format='pdf',dpi=200,bbox_inches='tight')

PATM = pd.DataFrame(np.array([adataP.X[:,adataP.var.index == 'GO_POST_ANAL_TAIL_MORPHOGENESIS'].reshape(-1),adataP.obs['disease'],adataP.obs['Annotation'],adataP.obs['SSA']]).T,index=adataP.obs.index,columns=['Post-Anal Tail Morphogenesis','Diagnosis','Annotation','SSA'])
PATM = PATM[PATM['Annotation'] == 'PRR4+CST3+WFDC2- SMACs']
fig = plt.figure(figsize=(5,5))
plt.clf()
sns.kdeplot(x=PATM['Post-Anal Tail Morphogenesis'].astype('float'),hue=PATM['Diagnosis'])
plt.savefig('scanpilot/kde_++-.pdf',format='pdf',dpi=200,bbox_inches='tight')
PATM['Clinical'] = PATM['Diagnosis'].replace('HV','nonPSS').astype('str') + PATM['SSA'].astype('str')
PATM['Clinical'] = PATM['Clinical'].replace('nonPSSNegative','nonPSS').replace('nonPSSPositive','nonPSS')
plt.clf()
sns.kdeplot(x=PATM['Post-Anal Tail Morphogenesis'].astype('float'),hue=PATM['Clinical'],hue_order=sorted(PATM['Clinical'].value_counts().index,key=str.lower),palette=['#1f77b4','#ff7f0e','#d62728'])
plt.savefig('scanpilot/kde_++-ssass.pdf',format='pdf',dpi=200,bbox_inches='tight')

adataP.obs['SSA'] = adataP.obs['SSA'].replace(True,'Positive').replace(False,'Negative').astype('category')
for i in celltypes:
	tempadata = adataP[adataP.obs['Annotation'] == i]
	tempadata = tempadata[tempadata.obs['Sample'].map(metadata['Sex'].to_dict()) == 0] #necessary otherwise XIST is a major difference
	sc.tl.rank_genes_groups(tempadata,'SSA',groups=['Positive'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/index/ssaix_'+i+'_1.pdf',format='pdf',dpi=200,bbox_inches='tight')
	sc.tl.rank_genes_groups(tempadata,'SSA',groups=['Negative'])
	plt.clf()
	sc.pl.rank_genes_groups(tempadata, n_genes=25, sharey=False)
	plt.savefig('scanpilot/index/ssaix_'+i+'_2.pdf',format='pdf',dpi=200,bbox_inches='tight')

def index_suite(adata,clinfeat,Gindex,label):
	csr = adata.X[:,adata.var.index.get_indexer(Gindex)]
	DF = pd.DataFrame(csr.toarray(),index=adata.obs.index,columns=Gindex)
	adata.obs[label] = DF.apply(sps.zscore).sum(axis=1)
	plt.clf()
	sns.barplot(data=adata.obs,x='Annotation',hue=clinfeat,y=label)
	plt.xticks(rotation=45,ha='right')
	plt.savefig('scanpilot/index/'+label+'_'+clinfeat+'_barplot.png',format='png',dpi=200,bbox_inches='tight')
	plt.close()
	uniq = adata.obs[clinfeat].unique()
	for i in range(len(uniq)):
		plt.clf()
		sc.pl.umap(adata[adata.obs[clinfeat] == uniq[i]],color=label)
		plt.savefig('scanpilot/index/'+label+'_'+clinfeat+'_umap_'+str(uniq[i])+'.png',format='png',dpi=200,bbox_inches='tight')
		plt.close()

adataP.obs['FS'] = adataP.obs['Sample'].map((metadata['Focus Score'] > 0).to_dict())
adataP.obs['SSA'] = adataP.obs['Sample'].map((metadata['SSA'] > 0).to_dict())
adataP.obs['SSAFS'] = adataP.obs['SSA'].astype('str') + '-' + adataP.obs['FS'].astype('str')
for clinfeat in ['SSA','FS','SSAFS','Disease']:
	for label in indexD.keys():
		index_suite(adataP,clinfeat,indexD[label],label.replace('/','|'))

plt.clf()
sns.heatmap(adataP.var.loc[list(indexD.keys())].corr())
plt.savefig('scanpilot/index/heatmap.png',format='png',dpi=200,bbox_inches='tight')

plt.clf()
sc.pl.dotplot(adataSG,groupby='leiden',var_names=['IFNAR1','IFNAR2','IFNGR1','IFNGR2','IFNLR1','IL10RB'])
plt.savefig('scanpilot/index/dotplot.png',format='png',dpi=200,bbox_inches='tight')

# THE T CELL CHECK

adataT = process_adata(adataSG[adataSG.obs['leiden'] == 'T Cells'],'SG_T')
adataT.write('adataT.h5ad')
plot_umap(adataT,['GNLY','GZMA','GZMB','GZMK','CD8A','CD3G','CD4'],'SGT_genes1')
plot_umap(adataT,['EOMES','PDCD1','ITGA1','SPRY1','CD40LG','CD24','IL2RA','FOXP3'],'SGT_genes2')
plot_umap(adataT,'SSA','SGT_SSA',palette={'Positive':'#e41a1c','Negative':'#4daf4a'})
plot_umap(adataT,'Disease','SGT_SjD',palette={'nonPSS':'#1f77b4','PSS':'#ff7f0e'})
plt.clf()
sc.pl.dotplot(adataT,['GNLY','GZMA','GZMB','GZMK','CD8A','CD4','EOMES','PDCD1','ITGA1','SPRY1','CD24','FOXP3'],groupby='Disease')
plt.savefig('scanpilot/SGT_stackedviolin.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.dotplot(adataT,['GNLY','GZMA','GZMB','GZMK','CD8A','CD4','EOMES','PDCD1','ITGA1','SPRY1','CD24','FOXP3'],groupby='SSA')
plt.savefig('scanpilot/SGT_stackedviolin_ssa.pdf',format='pdf',dpi=200,bbox_inches='tight')
adataT.obs['Clinical'] = adataT.obs['Disease'].replace('HV','nonPSS').astype('str') + adataT.obs['SSA'].astype('str')
adataT.obs['Clinical'] = adataT.obs['Clinical'].replace('nonPSSNegative','nonPSS').replace('nonPSSPositive','nonPSS')
plt.clf()
sc.pl.dotplot(adataT,['GNLY','GZMA','GZMB','GZMK','CD8A','CD4','EOMES','PDCD1','ITGA1','SPRY1','CD24','FOXP3'],groupby='Clinical',categories_order=('nonPSS','PSSNegative','PSSPositive'))
plt.savefig('scanpilot/SGT_stackedviolin_ssass.pdf',format='pdf',dpi=200,bbox_inches='tight')
adataPlasma = process_adata(adataSG[adataSG.obs['leiden'].isin(['IgA Plasma Cells','IgG Plasma Cells'])],'SG_plasma')
plot_umap(adataPlasma,['IGKC','IGLC2','IGHA1','IGHM','IGHD','IGHG1','AURKAIP1','TNFRSF4'],'SGP_genes1')
plt.clf()
sc.pl.dotplot(adataPlasma,['IGKC','IGLC2','IGHA1','IGHM','IGHD','IGHG1','AURKAIP1','TNFRSF4'],groupby='SSA')
plt.savefig('scanpilot/SGP_stackedviolin.png',format='png',dpi=200,bbox_inches='tight')
#AURKAIP1 TNFRSF4 half a mega upstream of aurk

# TRAJECTORY ANALYSIS
adataSGo = adataSG[adataSG.obs['Annotation'].isin(['PRR4+CST3+WFDC2- SMACs','PRR4+CST3+WFDC2+ SMACs','PRR4-CST3-WFDC2- SMACs','PRR4+CST3-WFDC2- SMACs','PRR4- Transitioning SMACs','Ductal Progenitors','Ionocytes','High ZG16B SMACs','Ductal Cells'])]
startcell = adataSGo.obs.index[adataSGo.X[:,adataSGo.var.index == 'MKI67'].argmax()]
sc.pp.pca(adataSGo)
sc.pp.neighbors(adataSGo, n_pcs=15, n_neighbors=10)
sc.tl.diffmap(adataSGo,n_comps=10)

sc.external.tl.wishbone(adataSGo,start_cell=startcell,k=20,num_waypoints=300)
plt.clf()
sc.pl.umap(adataSGo,color=['trajectory_wishbone','branch_wishbone'])
plt.savefig('scanpilot/SG_wisbone.pdf',format='pdf',dpi=200,bbox_inches='tight')

pd.crosstab(adataSGo.obs['branch_wishbone'],adataSGo.obs['Disease'])
pd.crosstab(adataSGo.obs['branch_wishbone'],adataSGo.obs['leiden']=='SMACs #1')

wishtraj = adataSGo.obs['trajectory_wishbone']
wfdc2 = adataSGo.X[:,adataSGo.var.index.get_loc('WFDC2')].toarray().T[0]
cst3 = adataSGo.X[:,adataSGo.var.index.get_loc('CST3')].toarray().T[0]
muc7 = adataSGo.X[:,adataSGo.var.index.get_loc('MUC7')].toarray().T[0]
lyz = adataSGo.X[:,adataSGo.var.index.get_loc('LYZ')].toarray().T[0]
zg16b = adataSGo.X[:,adataSGo.var.index.get_loc('ZG16B')].toarray().T[0]
wishtraj,wfdc2,cst3,muc7,lyz,zg16b = zip(*sorted(zip(wishtraj,wfdc2,cst3,muc7,lyz,zg16b)))
norm = lambda G : G/max(G.dropna())
plt.clf()
plt.plot(np.arange(len(wishtraj))/len(wishtraj),norm(pd.Series(wfdc2).rolling(1000,center=True,win_type='cosine').mean()),c='#377eb8',label='WFDC2 Expression')
plt.plot(np.arange(len(wishtraj))/len(wishtraj),norm(pd.Series(cst3).rolling(1000,center=True,win_type='cosine').mean()),c='#4daf4a',label='CST3 Expression')
plt.plot(np.arange(len(wishtraj))/len(wishtraj),norm(pd.Series(muc7).rolling(1000,center=True,win_type='cosine').mean()),c='#ff7f00',label='MUC7 Expression')
plt.plot(np.arange(len(wishtraj))/len(wishtraj),norm(pd.Series(lyz).rolling(1000,center=True,win_type='cosine').mean()),c='#ffff33',label='LYZ Expression')
plt.plot(np.arange(len(wishtraj))/len(wishtraj),norm(pd.Series(zg16b).rolling(1000,center=True,win_type='cosine').mean()),c='#984ea3',label='ZG16B Expression')
plt.legend()
plt.savefig('scanpilot/SG_wisbone_genes.pdf',format='pdf',dpi=200,bbox_inches='tight')

adataSGo[adataSGo.obs['leiden']=='SMACs #1']

# Ig FRACTIONS

plt.clf()
sc.pl.dotplot(adataSG,['IGHG1','IGHG2','IGHG3','IGHA1','IGHA2','IGHE','IGHD','IGHM'],'Disease')
plt.savefig('scanpilot/IgG_SG_dot.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.dotplot(adataPBMC,['IGHG1','IGHG2','IGHG3','IGHA1','IGHA2','IGHE','IGHD','IGHM'],'Disease')
plt.savefig('scanpilot/Ig_PBMC_dot.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.dotplot(adata,['IGHG1','IGHG2','IGHG3','IGHA1','IGHA2','IGHE','IGHD','IGHM'],'Disease')
plt.savefig('scanpilot/Ig_total_dot.png',format='png',dpi=200,bbox_inches='tight')

plt.clf()
scv.pl.proportions(adataSG,groupby='Annotation')
plt.savefig('scanpilot/SG_velocity_proportions.png',format='png',dpi=600,bbox_inches='tight')
plt.clf()
scv.pl.velocity_embedding_stream(adataSG,color='Annotation',legend_loc='right margin',arrow_style='->')
plt.savefig('scanpilot/SG_umap_velocitystream.png',format='png',dpi=600,bbox_inches='tight')
plt.clf()
scv.tl.recover_dynamics(adataSG)
scv.tl.latent_time(adataSG)
scv.pl.scatter(adataSG, color='latent_time', color_map='gnuplot', size=80)
plt.savefig('scanpilot/SG_velocity_latent.png',format='png',dpi=600,bbox_inches='tight')

# FLOW
cross = pd.crosstab(A.obs['Annotation'],A.obs['Sample']).T
cross = cross.divide(cross.sum(axis=1),axis=0)
props.index = props.index.tolist()
props.columns = props.columns.tolist()
props['WUS'] = props.index.map(metadata['WUS'].to_dict()).astype('float')
props['WSSF'] = props.index.map(metadata['WSSF'].to_dict()).astype('float')
plt.clf()
plt.figure(figsize=(10,10))
sns.heatmap(props.corr())
plt.savefig('associated_heatmap.pdf',format='pdf',dpi=200,bbox_inches='tight')

# METADATA PCA

samples = adataSG.obs['Sample'].value_counts().index
metadata = pd.read_csv('METADATA_scRNAseq_MSG.csv',index_col=0).loc[samples]
metadata = metadata.drop(['WUS','WSSF'],axis=1)
pca = skd.PCA()
exvar = pca.fit(metadata).explained_variance_ratio_
metaPCA = pca.fit_transform(metadata)
disease = samples.map(conditionD)
plt.clf()
plt.scatter(metaPCA[disease == 'nonPSS'].T[0],metaPCA[disease == 'nonPSS'].T[1],c='#377eb8',label='non-SjD')
plt.scatter(metaPCA[disease == 'PSS'].T[0],metaPCA[disease == 'PSS'].T[1],c='#e41a1c',label='SjD')
plt.scatter(metaPCA[disease == 'HV'].T[0],metaPCA[disease == 'HV'].T[1],c='#4daf4a',label='HV')
plt.legend()
plt.xlabel('Principal Component #1 (Explains '+str(int(exvar[0]*1000)/10.0)+'% Variance)')
plt.ylabel('Principal Component #2 (Explains '+str(int(exvar[1]*1000)/10.0)+'% Variance)')
plt.savefig('metadata_pca.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
M = metadata.drop('SSB',axis=1)
M = M.divide(M.max())
disease_colors = pd.Series(disease,index=samples).replace('nonPSS','#1f77b4').replace('PSS','#ff7f0e').replace('HV','#4daf4a')
sns.clustermap(M,row_colors=disease_colors,row_cluster=False,col_cluster=False)
plt.savefig('meta.pdf',format='pdf',dpi=200,bbox_inches='tight')

plt.clf()
plt.scatter(metaPCA[metadata['SSA'] > 0].T[0],metaPCA[metadata['SSA'] > 0].T[1],c='#e41a1c',label='SSA$^+$')
plt.scatter(metaPCA[metadata['SSA'] == 0].T[0],metaPCA[metadata['SSA'] == 0].T[1],c='#4daf4a',label='SSA$^-$')
plt.legend()
plt.xlabel('Principal Component #1 (Explains '+str(int(exvar[0]*1000)/10.0)+'% Variance)')
plt.ylabel('Principal Component #2 (Explains '+str(int(exvar[1]*1000)/10.0)+'% Variance)')
plt.savefig('metadata_pca_ssa.pdf',format='pdf',dpi=200,bbox_inches='tight')

adataSG.obs['Index Clusters'] = adataP.obs['leiden']
plot_umap(adataSG,'Index Clusters','SGindexcolors')


