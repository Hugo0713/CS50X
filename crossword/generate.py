import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in list(self.domains.keys()):
            for word in list(self.domains[var]):
                if len(word) != var.length:
                    self.domains[var].remove(word)
        # raise NotImplementedError

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]
        if overlap == None:
            return False
        flag_revision = False
        for word_x in list(self.domains[x]):
            flag = False
            for word_y in list(self.domains[y]):
                if word_x[overlap[0]] == word_y[overlap[1]]:
                    flag = True
                    break
            if flag == False:
                self.domains[x].remove(word_x)
                flag_revision = True
        return flag_revision

        # raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # if arcs == None:
        #     for x in list(self.domains.keys()):
        #         for y in list(self.domains.keys()):
        #             self.revise(x, y)
        # else:
        #     for arc in arcs:
        #         self.revise(arc[0], arc[1])
        # for var in list(self.domains.keys()):
        #     if len(self.domains[var]) == 0:
        #         return False
        # return True
        """
        use queue because the revise could have impact on other variables which should be revised maybe again
        """
        if arcs == None:
            queue = [(x, y) for x in self.domains for y in self.domains if x != y]
        else:
            queue = list(arcs)
        while queue:
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in list(self.domains):
                    if z != x and z != y:
                        queue.append((z, x))
        return True


        # raise NotImplementedError

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in list(self.domains):
            if var not in assignment:
                return False
        return True
        # raise NotImplementedError

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        if not self.assignment_complete(assignment):
            return True
        for var in assignment:
            if len(assignment[var]) != var.length:
                return False
            for neighbor in self.crossword.neighbors(var):
                overlap = self.crossword.overlaps[var, neighbor]
                if assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False
        return True
        # raise NotImplementedError

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        order = {word : 0 for word in self.domains[var]}
        neighbors = self.crossword.neighbors(var)
        for word in self.domains[var]:
            for neighbor in neighbors:
                if neighbor not in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    for word_neighbor in self.domains[neighbor]:
                        if word[overlap[0]] != word_neighbor[overlap[1]]:
                            order[word] += 1
        return sorted(order, key = order.get)
        # raise NotImplementedError

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = [var for var in self.domains if var not in assignment]
        mrv = min(unassigned_variables, key=lambda var: len(self.domains[var]))
        mrv_variables = [var for var in unassigned_variables if len(self.domains[var]) == len(self.domains[mrv])]
        if len(mrv_variables) == 1:
            return mrv_variables[0]
    
        degree = lambda var: sum(1 for neighbor in self.crossword.neighbors(var) if neighbor not in assignment)
        return max(mrv_variables, key=degree)
        # raise NotImplementedError

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.consistent(assignment) and self.assignment_complete(assignment):
            return assignment
        unassigned_variable = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(unassigned_variable, assignment):
            assignment[unassigned_variable] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != None:
                    return result
            del assignment[unassigned_variable]
        return None
        # raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
