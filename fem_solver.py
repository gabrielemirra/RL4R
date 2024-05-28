import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon, Point, LinearRing, MultiLineString
from shapely.plotting import plot_points, plot_line, plot_polygon
from collections import OrderedDict, defaultdict
import openseespywin.opensees as ops

class FEM():
    def __init__(self):
        
        self.placed_components = None
        self.nodal_displacements = [0]
        self.nodes, self.members, self.supports = None, None, None
        self.substructures_count = 0

    def solve_intersection(self, test_line, line_set):
        #returns a set of new lines if the intersection exists, otherwise the original line
        coords = [test_line.coords[0], test_line.coords[-1]]
        points = []
        no_int_flag = 1 # changes value if no intersections are recorded, meaning that the line is disconnected
        for i in range(len(line_set)):
            #print('intersection event: ', test_line.intersects(line_set[i]), test_line.intersection(line_set[i]))
            if test_line.intersects(line_set[i]) == True:
                no_int_flag = 0
                if test_line.intersection(line_set[i]).coords[0] not in coords: #check if the intersection doesn't happen at the end points
                    points.append(test_line.intersection(line_set[i]).coords[0])
        
        if len(points) > 0: #only when an intersection point is found on a line (not at the extremes), returns two segments
            # Add the coords from the points
            coords += points
            # Calculate the distance along the line for each point
            dists = [test_line.project(Point(p)) for p in coords]
            # sort the coords based on the distances
            coords = [p for (d, p) in sorted(zip(dists, coords))]
            # generate the Lines
            lines = [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]
        else:
            lines = [test_line]
        #     lines = [test_line] if no_int_flag == False else [None] 
        # if no_int_flag == 1:
        #     print('********************* the line is not connected ***************************')
        return lines


    def post_process(self):
        post_processed_components = []
        print('############ total placed components: ', len(self.placed_components))
        for i in range(len(self.placed_components)):
            placed_components_copy = self.placed_components[:]
            placed_components_copy.pop(i)
            post_processed_component = self.solve_intersection(self.placed_components[i], placed_components_copy)

            # if type(post_processed_component) == list:
            #     [plot_line(i,color='red') for i in post_processed_component]
            # else:
            #     plot_line(post_processed_component,color='red')
            post_processed_components.extend(post_processed_component)
            # if a line is not connected or intersects other lines (meaning that both end points (nodes) are unique top that line, remove the line)
        return MultiLineString([i for i in post_processed_components if i != None and i.length!=0])


    # Function to find connected lines
    def find_connected_lines(self, multiline):
        multilines = []
        endpoints_to_lines = defaultdict(list)
        for line in multiline.geoms:  # Iterate over individual LineStrings
            endpoints_to_lines[line.coords[0]].append(line)
            endpoints_to_lines[line.coords[-1]].append(line)

        def get_connected_lines(start_line, visited):
            stack = [start_line]
            connected = []
            while stack:
                line = stack.pop()
                if line not in visited:
                    visited.add(line)
                    connected.append(line)
                    start, end = line.coords[0], line.coords[-1]
                    neighbors = endpoints_to_lines[start] + endpoints_to_lines[end]
                    stack.extend(neighbors)
            return connected

        all_lines = list(multiline.geoms)  # List of individual LineStrings
        visited = set()
        connected_groups = []

        for line in all_lines:
            if line not in visited:
                connected_lines = get_connected_lines(line, visited)
                connected_groups.append(connected_lines)

        for group in connected_groups:
            multilines.append(MultiLineString(group))
        return multilines
        

    def filter_multilines_by_nodes(self, multilines, nodes):
        filtered_multilines = []
        for multiline in multilines:
            
            for line in multiline.geoms:

                if any(tuple(node) in list(line.coords) for node in nodes):
                    print('yes')
                    filtered_multilines.append(multiline)
                    break
        return filtered_multilines

    def model_builder(self, multiline_object):
        # Extract all points from the MultiLineString
        all_points = np.concatenate([np.array(line.coords) for line in multiline_object.geoms])

        # Find unique points and assign indices
        unique_points, indices = np.unique(all_points, axis=0, return_index=True)
        unique_points = unique_points[np.argsort(indices)]

        # Create a dictionary to map points to indices
        point_index_map = OrderedDict()
        for idx, point in enumerate(unique_points):
            point_index_map[tuple(point)] = idx+1

        # Create line connections based on indices
        line_connections = [[point_index_map[tuple(line.coords[0])], point_index_map[tuple(line.coords[-1])]] for line in multiline_object.geoms]

        return unique_points, line_connections

    def model_preview(self, nodes, members, supports=None):
        fig = plt.figure()
        axes = fig.add_axes([0.1,0.1,0.8,0.8])
        fig.gca().set_aspect('equal', adjustable='box')

        #Plot members
        for mbr in members:
            node_i = mbr[0] #Node number for node i of this member
            node_j = mbr[1] #Node number for node j of this member

            ix = nodes[node_i-1,0] #x-coord of node i of this member
            iy = nodes[node_i-1,1] #y-coord of node i of this member
            jx = nodes[node_j-1,0] #x-coord of node j of this member
            jy = nodes[node_j-1,1] #y-coord of node j of this member

            axes.plot([ix,jx],[iy,jy],'b') #Member

        #Plot nodes ########################################################################################################
        for i, n in enumerate(nodes):
            color = 'r' if n.tolist() in supports else 'b'
            axes.plot([n[0]],[n[1]], color=color, marker='o')
            axes.annotate(i+1, (n[0],n[1]))

        axes.set_xlabel('Distance (m)')
        axes.set_ylabel('Distance (m)')
        axes.set_title('Structure to analyse')
        axes.grid()
        plt.show()

    def deflected_model_preview(self, nodes, members):
        fig = plt.figure()
        axes_2 = fig.add_axes([0.1,0.1,0.8,0.8])
        fig.gca().set_aspect('equal', adjustable='box')

        xFac=1

        #Plot members
        for mbr in members:
            node_i = int(mbr[0]) #Node number for node i of this member
            node_j = int(mbr[1]) #Node number for node j of this member

            ix = nodes[node_i-1,0] #x-coord of node i of this member
            iy = nodes[node_i-1,1] #y-coord of node i of this member
            jx = nodes[node_j-1,0] #x-coord of node j of this member
            jy = nodes[node_j-1,1] #y-coord of node j of this member

            axes_2.plot([ix,jx],[iy,jy],'grey', lw=0.75) #Member

            ux_i = ops.nodeDisp(node_i, 1) #Horizontal nodal displacement
            uy_i = ops.nodeDisp(node_i, 2) #Vertical nodal displacement
            ux_j = ops.nodeDisp(node_j, 1) #Horizontal nodal displacement
            uy_j = ops.nodeDisp(node_j, 2) #Vertical nodal displacement

            axes_2.plot([ix + ux_i*xFac, jx + ux_j*xFac], [iy + uy_i*xFac, jy + uy_j*xFac],'r') #Deformed member

        axes_2.set_xlabel('Distance (m)')
        axes_2.set_ylabel('Distance (m)')
        axes_2.set_title('Deflected shape')
        axes_2.grid()
        plt.show()

    def analyse(self, nodes, members, supports=None): #support to be provided as node indices
        #Constants
        E = 200*10**9 #(N/m^2)
        A = 0.005 #(m^2)
        Iz = 0.00000199    
        #Remove any existing model
        ops.wipe()
        self.nodal_displacements = [0]
        #Set the model builder - 2 dimensions and 2 degrees of freedom per node
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        for i, n in enumerate(nodes):
            ops.node(i+1, float(n[0]), float(n[1]))

        ops.uniaxialMaterial("Elastic", 1, E)


        transfType='Linear'
        transfTag=1
        ops.geomTransf(transfType,transfTag)
        for i, mbr in enumerate(members):
            ops.element("elasticBeamColumn", i+1, int(mbr[0]), int(mbr[1]), A, E, Iz, transfTag)

        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)

        # Set boundary condition
        print(f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   {nodes}')
        supports_count = 0
        for ind, node in enumerate(nodes):
            print("############ ", node, type(node))
            print("************ ", supports)
            if node.tolist() in supports:
                ops.fix(ind+1,1,1,1)
                print(f'supports assigned to node no. {ind+1}')
                supports_count += 1
            else:
                ops.load(ind+1, 0.0, -5000.0, 0) #third component for rotation
                print(f'load assigned to node no. {ind+1}')
            
            # ops.fix(ind+1,1,1,1)
            # ops.load(ind+1, 0.0, -5000.0, 0) #third component for rotation
        if supports_count < len(nodes): #if at least one node is not a support
            #Analysis
            ops.system("BandSPD")
            ops.numberer("RCM")
            ops.constraints("Plain")
            ops.integrator("LoadControl", 1.0)
            ops.algorithm("Linear")
            ops.analysis("Static")
            ops.analyze(1)

            #Results
            #Nodal displacements
            for i, n in enumerate(nodes):
                ux = round(ops.nodeDisp(i+1, 1),5) #Horizontal nodal displacement
                uy = round(ops.nodeDisp(i+1, 2),5) #Vertical nodal displacement
                self.nodal_displacements+=[ux,uy]
                ##################################################################################### consider producing a node displacement map

    
            #print(f'Node {i+1}: Ux = {ux} m, Uy = {uy}')    
        
        #self.deflected_model_preview(nodes, members)

    def run(self, placed_components, supports):
        self.placed_components = placed_components
        self.supports = supports
        
        post_processed_components = self.post_process()

        ######################## group connected lines (result: list of multiline strings
        substructures = self.find_connected_lines(post_processed_components)
        self.substructures_count = len(substructures)
        print(f'found {len(substructures)} substructures')
        ######################## check if supports are included each polyline. rebuild multiline and perform analysis

        anchored_substructures= self.filter_multilines_by_nodes(substructures, self.supports)
        print('anchored_substructures: ',anchored_substructures)
        geometric_model  = MultiLineString([line for multiline in anchored_substructures for line in multiline.geoms])
        print('geometric model: ',geometric_model)
        # for structure in substructures:
        #     self.nodes, self.members = self.model_builder(structure)
        #     self.analyse(self.nodes, self.members, self.supports)
        if geometric_model.is_empty:
            pass
        else:
            self.nodes, self.members = self.model_builder(geometric_model)

            #self.supports must be actual coordinates, not indices
            self.analyse(self.nodes, self.members, self.supports)


    def visualise_model(self, view_deflected=True):
        if view_deflected:
            self.deflected_model_preview(self.nodes, self.members)
        else:
            self.model_preview(self.nodes, self.members, self.supports)


