import cv2
import numpy as np

# storing settings for semicircle
class SemiCircle:
    def __init__(
        self, thickness=10, color=(255, 0, 0), radius=100, 
        center=(250, 250), angle=0, start_angle=180, end_angle=360
        ):
        self.thickness = thickness
        self.color = color
        self.radius = (radius, radius)
        self.center = center
        self.angle = angle
        self.start_angle = start_angle
        self.end_angle = end_angle

class Line:
     def __init__(self, thickness=2, color=(0, 0, 0)):
        self.thickness = thickness
        self.color = color

def generate_points(radius, length, center, num_points):
    # center points
    cx, cy = center

    # generating points on circle
    outer_circle_points = [(radius * np.cos(i), radius * np.sin(i)) for i in np.linspace(np.pi, 2*np.pi, num_points)]

    inner_radius = radius - length
    inner_circle_points = [(inner_radius * np.cos(i), inner_radius * np.sin(i)) for i in np.linspace(np.pi, 2*np.pi, num_points)]


    # genrating point for drawing line using cv2, start_points and end_points
    start_points = [(int(cx + i), int(cy + j)) for i, j in outer_circle_points]
    end_points = [(int(cx + i), int(cy + j)) for i, j in inner_circle_points]
    return zip(start_points, end_points)

class Meter:
    def __init__(self, center, radius, circle_color):
        self.center = center
        self.radius = radius
        self.circle_color = circle_color

    def draw_meter(self, image, idx):
        # drawing semicircle
        circle = SemiCircle(center=self.center, radius=self.radius, color=self.circle_color)

        cv2.ellipse(
            image, circle.center, circle.radius, 
            circle.angle, circle.start_angle, circle.end_angle, 
            circle.color, circle.thickness
        )

        # drawing smaller fine lines
        line = Line(thickness=circle.thickness//3)

        for start, end in generate_points(self.radius - 10, self.radius * 0.05, self.center, 50):
            cv2.line(image, start, end, line.color, line.thickness)

        # drawing bigger lines
        line2 = Line(thickness=circle.thickness//2, color=(238, 222, 23))

        for start, end in generate_points(self.radius - 10, self.radius * 0.15, self.center, 10):
            cv2.line(image, start, end, line2.color, line2.thickness)

        # drawing needle anchor point
        cv2.circle(image, circle.center, 15, (0, 0, 0), -1)

        # emotion classes
        emotions = ['neutral', 'happy', 'surprise', 'sad', 'angry', 
                    "disgust", 'fear', 'contempt', 'unknown', 'NotFace']

        # points where text will be written
        pts = generate_points(self.radius*1.55, self.radius * 0.15, self.center, 10)
        pts = list(pts)

        for i, emot in enumerate(emotions):
            x, y = pts[i][0]
            x -= 20

            color = (98, 65, 255) if (i == idx) else (144,238,144)
            
            cv2.putText(
                image, emot, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 
                cv2.LINE_AA
            )           

        # needle 12, 178, 33
        line3 = Line(thickness=circle.thickness//2, color=(0, 0, 255))
        pts2 = generate_points(self.radius*0.7, self.radius*0.7, self.center, 10)
        pts2 = list(pts2)
        start, end = pts2[idx]

        cv2.line(image, start, end, line3.color, line3.thickness)

if __name__ == '__main__':
    image = np.ones((500, 500, 3))

    meter = Meter((250, 250), 150, (80, 127, 255))

    meter.draw_meter(image, 4)

    cv2.imshow('image', image)

    if cv2.waitKey(0) & 0xFF == 27:
            pass
    cv2.destroyAllWindows()