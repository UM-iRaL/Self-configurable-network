import itertools

# from main import HEIGHT, WIDTH, RADIUS

class ObservationModel(object):
    '''
    The ObservationModel class models the observable area in an area coverage problem.
    '''

    def __init__(self, radius, height, width, res):
        '''
        Constructs the Model.
        :param radius: Assumes a circle/square observable area with width and height given by range.
        '''
        self.height = height
        self.width = width
        self.res = res
        self.valid_grid_points = set([pt for pt in itertools.product(range(0, width+res, res), range(0, height+res, res))])
        # self.points = [point for point in itertools.product(np.arange(-radius, radius+res, res), repeat=2)] # square fov
        self.points = self.circle_fov(radius, res) # circle fov

    def get_observed_points(self, state, return_all=False):
        '''
        Given an input state, returns the observed points.
        :param state: The state observing from.
        :return: The set of observed points.
        '''
        # TODO Make this return a set of observed points that excludes any points which are not in the valid map area.
        all_obs_pts = set([self.motion_model(state, point) for point in self.points])
        
        if return_all:
            return all_obs_pts
        return all_obs_pts.intersection(self.valid_grid_points)

    def motion_model(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def circle_fov(self, radius, res):
        '''
        Returns the observed points in a circle field-of-view.
        :param radius: The sensing radius.
        :param res: The resolution of observable points.
        :return: The set of observed points.
        '''
        points = set()
        for x, y in itertools.product(range(-radius, radius+res, res), range(-radius, radius+res, res)):            
            if x**2 + y**2 <= radius**2:
                # print((x,y))
                points.add((x, y))
                # yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))
        return points


# if __name__ == "__main__":
#     obs = ObservationModel()
    # print(len(obs.valid_grid_points))
    # print(type(obs.valid_grid_points))
    # print(type(obs.points))
    # print(len(obs.points))
    # for pt in obs.valid_grid_points:
    #     print(pt)
    # for pt in obs.points:
    #     print(pt)
