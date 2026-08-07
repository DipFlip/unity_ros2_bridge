from unity_ros2_bridge.occupancy_grid import RaytracedOccupancyGrid


def test_ray_marks_free_cells_and_occupied_endpoint():
    grid = RaytracedOccupancyGrid(1.0, 10, 10, 0.0, 0.0)

    grid.update_ray(1.5, 1.5, 4.5, 1.5)
    origin_x, origin_y, width, height, data = grid.cropped_data(0)

    assert (origin_x, origin_y, width, height) == (1.0, 1.0, 4, 1)
    assert list(data) == [0, 0, 0, 100]


def test_repeated_misses_clear_an_old_hit():
    grid = RaytracedOccupancyGrid(1.0, 10, 10, 0.0, 0.0)
    grid.update_ray(1.5, 1.5, 3.5, 1.5)

    for _ in range(4):
        grid.update_ray(1.5, 1.5, 4.5, 1.5)

    _, _, _, _, data = grid.cropped_data(0)
    assert list(data) == [0, 0, 0, 100]


def test_crop_keeps_unobserved_padding_unknown():
    grid = RaytracedOccupancyGrid(1.0, 10, 10, 0.0, 0.0)
    grid.update_ray(4.5, 4.5, 4.5, 4.5)

    origin_x, origin_y, width, height, data = grid.cropped_data(1)

    assert (origin_x, origin_y, width, height) == (3.0, 3.0, 3, 3)
    assert list(data) == [-1, -1, -1, -1, 100, -1, -1, -1, -1]
