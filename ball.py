from math import dist
def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2
def max_radius(points):
    points = sorted(points)
    
    k = 5
    spheres = []
    
    for i in range(1, min(k, len(points))):
        spheres.append((dist(points[0], points[i]) / 2, 
                        [points[0][0], points[i][0]],
                        [points[0][1], points[i][1]],
                        [points[0][2], points[i][2]]))      
        
    used = [0] * len(points)  
    
    
    def dist_to_spheres(point, spheres):
        min_dist = float('inf')
        for s in spheres:
            x_dist = min((point[0] - x)**2 for x in s[1])
            y_dist = min((point[1] - y)**2 for y in s[2])
            z_dist = min((point[2] - z)**2 for z in s[3])
            dist = x_dist + y_dist + z_dist
            min_dist = min(min_dist, dist - s[0])
        return min_dist
    
    for i in range(k, len(points)):
        min_dist = dist_to_spheres(points[i], spheres)
        
        for s in spheres:
            s[0] = max(s[0], min_dist / 2.0)  
            s[1].append(points[i][0])
            s[2].append(points[i][1])
            s[3].append(points[i][2])
            
        used[i] = 1
    return [s[0] for s in spheres]
print(max_radius([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))