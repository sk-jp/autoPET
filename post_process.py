import cc3d
import copy


def post_process(input, num_classes):

    output = copy.copy(input)

    connectivity = 26    
    for n in range(1, num_classes):
        # extract connected regions
        label = cc3d.connected_components(input==n, connectivity=connectivity)
        stats = cc3d.statistics(label)
                        
        for m in range(2, len(stats['voxel_counts'])):
            # remove non-maximum regions
            output[label == m] = 0           
        
#        print('n:', n)
#        print(stats['voxel_counts'])

    return output
