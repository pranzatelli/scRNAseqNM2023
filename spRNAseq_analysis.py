import numpy as np
import scipy.stats as sps
from statsmodels.stats.multitest import multipletests
import pandas as pd
import scanpy as sc
import geojson
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
	plt.savefig(file+'_umap.pdf',format='pdf',bbox_inches='tight')

def plot_umap_labelson(A,file):
	plt.clf()
	sc.pl.umap(A,color='leiden', legend_loc='on data',legend_fontsize=5, title='')
	plt.savefig('umaplabelson_'+file+'.png',format='png',dpi=200,bbox_inches='tight')

def process_adata(A,label,res=1):
	sc.pp.highly_variable_genes(A, min_mean=0.0125, max_mean=3, min_disp=0.5)
	sc.tl.pca(A)
	sc.external.pp.bbknn(A)
	A.uns['neighbors']['params']['metric'] = 'cosine'
	sc.tl.umap(A)
	sc.tl.leiden(A,resolution=res)
	plot_umap(A,['pct_counts_mt','pct_counts_rp','n_genes','total_counts'],'qc_'+label)
	plot_umap(A,'Sample','samples_'+label)
	plot_umap(A,'disease','diagnosis_'+label)
	plot_umap(A,'leiden','leiden_'+label)
	sc.tl.rank_genes_groups(A,'leiden')
	plt.clf()
	sc.pl.rank_genes_groups(A, n_genes=25, sharey=False)
	plt.savefig('clustergenes_'+label+'.png',format='png',dpi=200,bbox_inches='tight')
	plt.close('all')
	return A

import cell2location
import scvi
import networkx as nx
from anndata import AnnData
import squidpy as sq

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

samplenames=["MSG","MSG2","MSG3","MSG4","MSG5","MSG6","MSG7","MSG8","MSG9","MSG10","MSG11","MSG12","MSG13","MSG14","MSG16","MSG17","MSG18","MSG19","MSG20","MSG21","MSG22","MSG23","MSG24","MSG25","MSG26","MSG29","MSG30","MSG31","PBMC","PBMC2","PBMC3","PBMC4","PBMC5","PBMC6","PBMC7","PBMC8","PBMC9","PBMC17","PBMC18","PBMC19","PBMC20","PBMC21","PBMC23","SP1","SP2","SP3","SP4"]
conditions=["PSS","nonPSS","PSS","PSS","PSS","PSS","nonPSS","nonPSS","nonPSS","PSS","nonPSS","PSS","nonPSS","nonPSS","nonPSS","PSS","PSS","nonPSS","PSS","nonPSS","Unknown","nonPSS","HV","HV","HV","nonPSS","PSS","HV","nonPSS","PSS","nonPSS","PSS","PSS","PSS","PSS","nonPSS","nonPSS","nonPSS","PSS","PSS","nonPSS","PSS","nonPSS","nonPSS","PSS","PSS","PSS"]
tissues=["SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","SG","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","PBMC","SG","SG","SG","SG"]
conditionD = {samplenames[i]:conditions[i] for i in range(len(samplenames))}
tissueD = {samplenames[i]:tissues[i] for i in range(len(samplenames))}
samples = [x.replace('/data/ChioriniCompCor/experiments/2021-01-20_scRNA_refactoring/','') for x in glob('/data/ChioriniCompCor/experiments/2021-01-20_scRNA_refactoring/samples/*')]
samples = [sample for sample in samples if 'MUC' not in sample]
samples = [sample for sample in samples if '-' not in sample]
samples = [sample for sample in samples if 'pSG' not in sample]
samples.remove('samples/MSG29');samples.remove('samples/MSG22');samples.remove('samples/SP4');samples.remove('samples/MSG6')
samples.remove('samples/MSG');samples.remove('samples/SP1');samples.remove('samples/MSG24')
def loadnorm_data_sc_nolog(sample):
	label = sample.split('/')[-1]
	adata = sc.read_10x_h5('/data/ChioriniCompCor/experiments/2021-01-20_scRNA_refactoring/'+sample+'/cellbender_filtered.h5')
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
	adata.obs['Sample'] = label
	return adata

def loadnorm_data_sp(sample):
	label = sample.split('/')[0]
	adata = sc.read_visium(sample)
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
	adata.obs['Sample'] = label
	barcode_pos = pd.read_csv(sample+'/spatial/tissue_positions_list.csv',index_col=0,header=None)
	adata.obs['array_row'] = adata.obs.index.map(barcode_pos[2])
	adata.obs['array_col'] = adata.obs.index.map(barcode_pos[3])
	return adata

adata = [loadnorm_data_sc_nolog(s) for s in samples if 'PBMC' not in s]
adata = adata[0].concatenate(adata[1:])

adata_ref = sc.read_h5ad('/data/ChioriniCompCor/experiments/2021-01-20_scRNA_refactoring/adataSG.h5ad')
adata_ref.X = adata.X
scvi.data.setup_anndata(adata=adata_ref,batch_key='Sample',labels_key='Annotation')
mod = cell2location.models.RegressionModel(adata_ref)
mod.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002, use_gpu=True)
plt.clf()
mod.plot_history(20)
plt.savefig('modhistory.png',format='png',dpi=200,bbox_inches='tight')

adata_ref = mod.export_posterior(adata_ref, sample_kwargs={'num_samples':1000,'batch_size':2500,'use_gpu':False})
mod.save("mod", overwrite=True)
adata_ref.write("mod/sc.h5ad")

adata_ref = sc.read_h5ad('mod/sc.h5ad')
mod = cell2location.models.RegressionModel.load('mod',adata_ref)

adata = parallelize(loadnorm_data_sp,glob('*-*')+glob('*_a/outs')+glob('*_b/outs'))
adata_vis = adata[0].concatenate(adata[1:])
diagnosis = pd.read_csv('METADATA_spRNAseq_MSG_unified.csv',dtype='str',index_col=19)['Diagnosis']
diagnosis = diagnosis[diagnosis.index.dropna()]
adata_vis.obs['Disease'] = adata_vis.obs['Sample'].map(diagnosis)
adata_vis = adata_vis[adata_vis.obs['Disease'] != 'ICI']
inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}' for i in adata_ref.uns['mod']['factor_names']]].copy()
intersect = np.intersect1d(adata_vis.var_names,inf_aver.index)
adata_vis = adata_vis[:,intersect].copy()
inf_aver = inf_aver.loc[intersect,:].copy()
scvi.data.setup_anndata(adata=adata_vis,batch_key='Sample')
mod = cell2location.models.Cell2location(adata_vis,cell_state_df=inf_aver,N_cells_per_location=10,detection_alpha=50)
mod.train(max_epochs=30000,batch_size=None,train_size=1,use_gpu=True)
adata_vis = mod.export_posterior(adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True})
adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']
plt.clf()
mod.plot_history(1000)
plt.savefig('spatialhistory.png',format='png',dpi=200,bbox_inches='tight')
mod.save("mod", overwrite=True)
plt.clf()
mod.plot_QC()
plt.savefig('modQC.png',format='png',dpi=200,bbox_inches='tight')
adata_vis.obs.columns = [c.replace('means_per_cluster_mu_fg_','') for c in adata_vis.obs.columns]
adata_vis.write("mod/sp.h5ad")

adata_vis = sc.read_h5ad('mod/sp.h5ad')

cellnames = [x.split('means_per_cluster_mu_fg_')[1] for x in adata_vis.uns['mod']['factor_names']]
cells = adata_vis.obs[cellnames]
cells = cells.divide(cells.sum(axis=1),axis=0)
adata_vis.obs[[x.split('means_per_cluster_mu_fg_')[1] for x in adata_vis.uns['mod']['factor_names']]] = cells
plt.clf()
sc.pl.spatial(adata_vis[adata_vis.obs['Sample'] == '3-1b'],cmap='magma',color=adata_ref.uns['mod']['factor_names'],ncols=6,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide_ss_ssa.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.spatial(adata_vis[adata_vis.obs['Sample'] == '1-1b'],cmap='magma',color=adata_ref.uns['mod']['factor_names'],ncols=6,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide_ss_nossa.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.spatial(adata_vis[adata_vis.obs['Sample'] == '3b_b'],cmap='magma',color=adata_ref.uns['mod']['factor_names'],ncols=6,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide2.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.spatial(adata_vis[adata_vis.obs['Sample'] == '1-1b'],cmap='magma',color=adata_ref.uns['mod']['factor_names'],ncols=6,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide3.png',format='png',dpi=200,bbox_inches='tight')

def beautiful_c2l_plot(sample,celltypes,label):
	plt.clf()
	adata = loadnorm_data_sp(sample)
	for celltype in celltypes:
		cellD = adata_vis.obs[celltype].to_dict()
		cellD = {'-'.join(key.split('-')[:-1]):cellD[key] for key in cellD.keys()}
		adata.obs[celltype] = adata.obs.index.map(cellD)
	with matplotlib.rc_context({'figure.figsize': (15, 15)}):
		cell2location.plt.plot_spatial(adata,color=celltypes,labels=celltypes,style='fast',max_color_quantile=0.992,circle_diameter=6,colorbar_position='right')
	plt.savefig('slide_'+label+'.png',format='png',dpi=200)

beautiful_c2l_plot('3-1b',['CD8+ Exhausted T Cells'],'SjD')
beautiful_c2l_plot('3-1a',['CD8+ Exhausted T Cells'],'nonSjD')

tiff = plt.imread('slide1_area C1_block2.tif')
gj1 = geojson.load(open('slide1_area C1_block2_sample a.geojson','r'))['features']
gj2 = geojson.load(open('slide1_area C1_block2_sample b.geojson','r'))['features']

plt.clf()
plt.imshow(tiff)
threeone = adata_vis[adata_vis.obs['Sample'].isin(['3-1a','3-1b'])]
xy = threeone.obsm['spatial'].T
plt.scatter(xy[0],xy[1],c=threeone.obs['PRR4+CST3+WFDC2- SMACs'],cmap='cool',s=0.2,alpha=0.5)

for ann in gj1+gj2:
	if 'classification' in ann['properties']:
		tissue = ann['properties']['classification']['name']
	if tissue == 'Immune cells':
		color = 'r'
	elif tissue == 'Ducts':
		color = 'b'
	else:
		color = 'g'
	if ann['geometry']['type'] == 'Polygon':
		xy = np.array(ann['geometry']['coordinates'][0]).T
		plt.plot(xy[0],xy[1],c=color,lw=0.3)
	elif ann['geometry']['type'] == 'MultiPolygon':
		xy = np.array(ann['geometry']['coordinates'][0][0]).T
		plt.plot(xy[0],xy[1],c=color,lw=0.3)

plt.colorbar(label='Proportion PRR4+CST3+WFDC2- SMACs - Cell2Location')
green_line = matplotlib.patches.Patch(color='green', label='Acini')
red_line = matplotlib.patches.Patch(color='red', label='Immune')
blue_line = matplotlib.patches.Patch(color='blue', label='Ducts')
plt.legend(handles=[green_line,red_line,blue_line],title='Manually Annotated Regions',fontsize='small')
plt.savefig('test_geojson_S.pdf',format='pdf',dpi=400,bbox_inches='tight')
plt.savefig('test_geojson_S.png',format='png',dpi=400,bbox_inches='tight')


plt.clf()
adata = loadnorm_data_sp('3d_b/outs')
clusters = ['Regulatory T Cells','Dendritic Cells']
slide = cell2location.utils.select_slide(adata_vis,'3a_b',batch_key='Sample')
with matplotlib.rc_context({'figure.figsize': (15, 15)}):
	cell2location.plt.plot_spatial(adata_vis,color=clusters,labels=clusters,show_img=True,style='fast',max_color_quantile=0.992,circle_diameter=6,colorbar_position='right')

plt.savefig('slide_healthy.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.spatial(adata_vis,cmap='magma',color=adata_ref.uns['mod']['factor_names'],ncols=2,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide4.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sc.pl.spatial(adata_vis,cmap='magma',color=['WFDC2','CST3','PRR4','MUC7','MUC5B','HLA-DRA','CD8A','CD4','FOXP3','GZMA','GZMB','GZMK'],ncols=3,size=1.3,img_key=None,spot_size=16,vmin=0,vmax='p99.2')
plt.savefig('slide5.png',format='png',dpi=200,bbox_inches='tight')
plt.clf()
sns.clustermap(cells.corr())
plt.savefig('0corr.pdf',format='pdf',dpi=300,bbox_inches='tight')

def corr1(A):
	G = nx.Graph()
	for sample in A.obs['Sample'].unique():
		sample_adata = A[A.obs['Sample'] == sample]
		for ix1 in range(len(sample_adata.obs.index)):
			node1 = sample_adata.obs.iloc[ix1]
			for ix2 in range(ix1):
				node2 = sample_adata.obs.iloc[ix2]
				if abs(node1['array_col'] - node2['array_col']) + abs(node1['array_row'] - node2['array_row']) < 3:
					G.add_edge(node1.name,node2.name)
		print("Done with "+sample)
	nx.write_graphml(G,'G.graphml')
	newD = {}
	e1 = cells.reindex([e[0] for e in nx.edges(G)]+[e[1] for e in nx.edges(G)])
	e2 = cells.reindex([e[1] for e in nx.edges(G)]+[e[0] for e in nx.edges(G)])
	for label1,content1 in e1.iteritems():
		newD[label1] = {}
		for label2,content2 in e2.iteritems():
			newD[label1][label2] = np.corrcoef(content1.values,content2.values)[0][1]
	return pd.DataFrame(newD)

plt.clf()
sns.clustermap(corr1(adata_vis))
plt.savefig('1corr.pdf',format='pdf',dpi=200,bbox_inches='tight')

corr1ss = corr1(adata_vis[adata_vis.obs['Disease'] == 'SS'])
corr1non = corr1(adata_vis[adata_vis.obs['Disease'] == 'nonSS'])
plt.clf()
clustergrid = sns.clustermap(corr1non)
plt.savefig('1corr_non.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
order = corr1ss.index[clustergrid.dendrogram_row.reordered_ind]
sns.heatmap(corr1ss[order].loc[order])
plt.savefig('1corr_ss.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
corr1diff = corr1ss-corr1non
sns.clustermap(corr1diff[order].loc[order])
plt.savefig('1corr_diff.pdf',format='pdf',dpi=200,bbox_inches='tight')

corr0ss = adata_vis[adata_vis.obs['Disease'] == 'SS'].obs[cellnames].corr() 
corr0non = adata_vis[adata_vis.obs['Disease'] != 'SS'].obs[cellnames].corr()
plt.clf()
clustergrid = sns.clustermap(corr0non)
plt.savefig('0corr_non.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
order = corr0ss.index[clustergrid.dendrogram_row.reordered_ind]
sns.heatmap(corr0ss[order].loc[order])
plt.savefig('0corr_ss.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
corr0diff = corr0ss-corr0non
sns.heatmap(corr0diff[order].loc[order])
plt.savefig('0corr_diff.pdf',format='pdf',dpi=200,bbox_inches='tight')

ssa = pd.read_csv('METADATA_spRNAseq_MSG_unified.csv',dtype='str',index_col=19)['SSA'].astype('float') > 0
ssa = ssa[ssa.index.dropna()]
adata_vis.obs['SSA'] = adata_vis.obs['Sample'].map(ssa).replace(False,'Negative').replace(True,'Positive')
corr0neg = adata_vis[adata_vis.obs['SSA'] == 'Negative'].obs[cellnames].corr() 
corr0pos = adata_vis[adata_vis.obs['SSA'] == 'Positive'].obs[cellnames].corr()
plt.clf()
sns.clustermap(corr0pos)
plt.savefig('0corr_pos.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
sns.clustermap(corr0neg)
plt.savefig('0corr_neg.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
sns.clustermap(corr0pos-corr0neg)
plt.savefig('0corr_ssadiff.pdf',format='pdf',dpi=200,bbox_inches='tight')

corr1ssa = corr1(adata_vis[adata_vis.obs['SSA'] == 'Positive'])
corr1neg = corr1(adata_vis[adata_vis.obs['SSA'] == 'Negative'])
plt.clf()
sns.clustermap(corr1ssa)
plt.savefig('1corr_ssa.pdf',format='pdf',dpi=300,bbox_inches='tight')
plt.clf()
sns.clustermap(corr1neg)
plt.savefig('1corr_neg.pdf',format='pdf',dpi=300,bbox_inches='tight')
plt.clf()
sns.clustermap(corr1ssa-corr1neg)
plt.savefig('1corr_ssadiff.pdf',format='pdf',dpi=300,bbox_inches='tight')

metadata = pd.read_csv('METADATA_spRNAseq_MSG_unified.csv',dtype='str',index_col=19)
adata_vis.obs['WUS'] = adata_vis.obs['Sample'].map(metadata['WUS'].to_dict()).astype('float')
adata_vis.obs['WSSF'] = adata_vis.obs['Sample'].map(metadata['WSSF'].to_dict()).astype('float')
plt.clf();plt.figure(figsize=(12,10));sns.heatmap(adata_vis.obs.corr());plt.savefig('test_heatmap.png',format='png',bbox_inches='tight')


def determine_cooc(cell):
	cell = cells.loc[cell]
	cooc = cell.apply(lambda y: ((y + cell) * np.power(np.e,(-3 * np.abs(cell - y)))) / 2)
	cooc = cooc.values[np.triu_indices(31,k=1)]
	return cooc

coocmtx = np.array(parallelize(determine_cooc,cells.index))
cooclabels = pd.Series(cells.columns).apply(lambda x: x + ' by ' + pd.Series(cells.columns)).values.reshape(-1)
cooclabels = cooclabels.reshape(31,31)[np.triu_indices(31,k=1)]

def corr2_coeff(A, B):
	#https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

genecooc = corr2_coeff(adata_vis.X.T.toarray(),coocmtx.T)
genecoocDF = pd.DataFrame(genecooc,index=adata_vis.var.index,columns=cooclabels).dropna()
adata_cooc = AnnData(X=coocmtx,obs=adata_vis.obs,var=genecoocDF.T)
plt.clf()
adata_cooc.obsm['spatial'] = adata_vis.obsm['spatial']
sc.pl.spatial(adata_cooc[adata_cooc.obs['Sample'] == '3-1a'],cmap='magma',color=['Ductal Cells by Fibroblasts','Ductal Cells by Ductal Progenitors','CD8+ Exhausted T Cells by PRR4+CST3+WFDC2- SMACs','High ZG16B SMACs by PRR4+CST3-WFDC2- SMACs'],ncols=2,size=1.3,img_key=None,spot_size=16,vmin=0,vmax=[0.3,0.225,0.07,0.011])
plt.savefig('slide6.pdf',format='pdf',dpi=200,bbox_inches='tight')
plt.clf()
adata_cooc.obsm['spatial'] = adata_vis.obsm['spatial']
sc.pl.spatial(adata_cooc[adata_cooc.obs['Sample'] == '3-1b'],cmap='magma',color=['Ductal Cells by Fibroblasts','Ductal Cells by Ductal Progenitors','CD8+ Exhausted T Cells by PRR4+CST3+WFDC2- SMACs','High ZG16B SMACs by PRR4+CST3-WFDC2- SMACs'],ncols=2,size=1.3,img_key=None,spot_size=16,vmin=0,vmax=[0.3,0.225,0.07,0.011])
plt.savefig('slide7.pdf',format='pdf',dpi=200,bbox_inches='tight')

sc.tl.rank_genes_groups(adata_cooc,'SSA')
plt.clf();sc.pl.rank_genes_groups_matrixplot(adata_cooc,n_genes=5)
plt.savefig('coocssa.pdf',format='pdf',dpi=200,bbox_inches='tight')
sc.tl.rank_genes_groups(adata_cooc,'Disease')
plt.clf();sc.pl.rank_genes_groups_matrixplot(adata_cooc,n_genes=5)
plt.savefig('coocdis.pdf',format='pdf',dpi=200,bbox_inches='tight')
adata_cooc.obs['Clinical'] = adata_cooc.obs['Disease'].replace('HV','nonSS').astype('str') + adata_cooc.obs['SSA']
adata_cooc = adata_cooc[adata_cooc.obs['Clinical'] != 'nanNegative'];adata_cooc = adata_cooc[adata_cooc.obs['Clinical'] != 'nanPositive']
sc.tl.rank_genes_groups(adata_cooc,'Clinical')
plt.clf();sc.pl.rank_genes_groups_matrixplot(adata_cooc,n_genes=5)
plt.savefig('cooclin.pdf',format='pdf',dpi=200,bbox_inches='tight')

colocs = list(adata_cooc.uns['rank_genes_groups']['names'].SS[:5]) + list(adata_cooc.uns['rank_genes_groups']['names'].HV[:5])
genes_to_look_at = []
for coloc in colocs:
	vals = genecoocDF[coloc].sort_values()
	genes_to_look_at += list((vals[:5]+vals[-5:]).index)

genes_to_look_at = list(set(genes_to_look_at))
plt.clf();sns.clustermap(genecoocDF[colocs].loc[genes_to_look_at].T,row_cluster=False,figsize=(10,5))
plt.savefig('coocheat.pdf',format='pdf',dpi=200,bbox_inches='tight')


