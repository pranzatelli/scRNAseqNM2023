library(CellChat)
library(Seurat)

#create a subset seurat object for each condition
#use the cell type labels as metadata
#utilize default parameters
cellchat <- createCellChat(object = ctrl, group.by = "celltype")
CellChatDB <- CellChatDB.human
CellChatDB.use <- CellChatDB
cellchat@DB <- CellChatDB.use
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat)
cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)
ctrlgroupSize <- as.numeric(table(cellchat@idents))
cellchatctrl <- cellchat

