## I am currently working on point cloud transformer ##
## The objective is to improve grab the locality of points in point cloud to find the important points and generate the attention map ##
from pytorch3d.ops import knn_points,knn_gather
import einops
from sparsemax import Sparsemax

class RelationalKNNSA(nn.Module):
            """ KNN Self Attention with Positional Embedding """
            """ Mostly based on Point Transformer V2: Grouped Vector Attention and Partition-based Pooling. ArXiv. /abs/2210.05666 """
            """ Using pytorch3d.ops for computing the neighbours in pytorch """
            """ Using Sparsemax for normalizing the attention map instead of softmax """
            def __init__(self, in_dim):
                super(RelationalKNNSA, self).__init__()

                self.query_conv = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.key_conv =  nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True)
                )

                self.value_conv =  nn.Linear(in_dim, in_dim, bias=True)

                self.linear = nn.Sequential(
                nn.Linear(3, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
                )

                # self.projection_conv =  nn.Linear(in_dim,in_dim)
                # nn.init.constant_(self.projection_conv.weight, 0)

                self.normalized = Sparsemax(dim=1)

                self.weight_encoding = nn.Sequential(
                    nn.Linear(in_dim, in_dim, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_dim, in_dim)
                )

                self.attn_drop = nn.Dropout(0.5)

            def get_batch_splits(self, me_tensor):
                _, counts = torch.unique(me_tensor.C[:, 0], return_counts=True)
                splits_C = torch.split(me_tensor.C, counts.tolist(), dim=0)
                splits_C = [s[:, 1:] for s in splits_C]
                splits_F = torch.split(me_tensor.F, counts.tolist())

                return splits_C, splits_F

            def pytorchknn(self,x):

                x = x.unsqueeze(0)
                k = min (16,x.size(1))
                knn = knn_points(x, x, K=k)
                indices = knn[1]
                neighbors = knn_gather(x,indices)
                return neighbors.squeeze(0),indices.squeeze(0),k # return gathered points indices

            def forward(self, x):

                scale = x.shape[1]
                Batch_split_coordinate, Batch_split_features = self.get_batch_splits(x)
                projection = []
                for i in range (len(Batch_split_features)):

                    feature_point = Batch_split_features[i].to(x.device)
                    coordinate_point = Batch_split_coordinate[i].to(x.device).float()

                    neighbors_coordinates, indices, k = self.pytorchknn(coordinate_point)
                    differential_coordinates = neighbors_coordinates - coordinate_point.unsqueeze(1)
                    neighbors_features = feature_point[indices]
                    feature_coordinate = self.linear(differential_coordinates)
                    positional_embeding= feature_coordinate + neighbors_features
                    Query = self.query_conv(Batch_split_features[i])
                    Key = self.key_conv(positional_embeding)
                    Value = self.value_conv(positional_embeding)
                    unsqueeze_query = Query.unsqueeze(1)
                    relation_qk = Key - unsqueeze_query
                    relation_qk = relation_qk * feature_coordinate
                    weight = self.weight_encoding(relation_qk)
                    weight = self.attn_drop(self.normalized(weight))

                    mask = torch.sign(indices + 1)
                    weight = torch.einsum("n s g, n s -> n s g", weight, mask)
                    value = einops.rearrange(Value, "n ns (g i) -> n ns g i", g=scale)
                    feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
                    feat = einops.rearrange(feat, "n g i -> n (g i)")
    
                    projection.append(feat)
                scores = torch.cat(projection, dim=0)
                attended_input = ME.SparseTensor(features=scores, tensor_stride=x.tensor_stride, device = x.device, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                attended_input = x + attended_input
                return attended_input
