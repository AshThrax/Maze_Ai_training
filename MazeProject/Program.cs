using MazeProject.ToDoItem;
using TorchSharp;

int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};

const string Up = "up";
const string Down = "down";
const string Left = "left";
const string Right = "right";

string[] actions = { Up, Down, Left, Right };
int[,] rewards;
const int wall_reward_value = -500;
const int floor_reward_value = -10;
const int goal_reward_value = 500;

void setupRewards(int[,] maze,int wallvalue, int floorvalue, int goalvalue)
{
    int mazerows=maze.GetLength(0);
    int mazeColumns =maze.GetLength(1); 
    rewards= new int [mazerows,mazeColumns];

    for (int i = 0; i < mazerows; i++)
    {
        for (int j = 0; j < mazeColumns; j++)
        {
            switch (maze[i, j])
            {
               case 0:
                    rewards[i, j] = wallvalue;
                    break;
               case 1:
                    rewards [i, j] = floorvalue;
                break;

               case 2:
                    rewards [i, j] =goalvalue;
                    break;
            }
        }
    }
}

torch.Tensor Qvalues;

//
void setupQvalues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns=maze.GetLength(1);

    Qvalues=torch.zeros(mazeRows,mazeColumns,4);

}
//determine if the Ai has touched a wall
bool hasHitWallOrEndOfMaze(int currentRow, int CurrentColumn, int floorValue)
{
    return rewards[currentRow,CurrentColumn] !=floorValue;
}
//determnine the next action that the Ai is capable to make basez on his position
long determinedenxtAction(int currentRow, int currentColumns, float epsilon)
{
    Random randomn = new Random();
    double randomBetween0and1=randomn.NextDouble();
    long nextAction = randomBetween0and1 < epsilon ? torch.argmax(Qvalues[currentRow, currentColumns]).item<long>() : randomn.Next(4);
    return nextAction;  
}
// Allow the Ai to move inside de maze
(int, int) moveOneSpace(int [,] maze, int currentRow, int currentColumns, long currentAction)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    int nextRow= currentRow;
    int nextCol= currentColumns;

    if (actions[currentAction] == Up && currentRow > 0)
    {
        nextRow--;
    }
    else if (actions[currentAction] == Down && currentRow < mazeRows - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == Left && currentColumns > 0)
    {
        nextCol--;
    }
    else if (actions[currentAction] == Right && currentColumns < mazeColumns - 1)
    {
        nextCol++;
    }
    return(nextRow, nextCol);
}
//train the model AI in order to be able to Go out of the maze
void trainModel(int[,] maze,int floorValue, float epsilon,float discountFactor, float learningRate, float episodes)
{
    for (int episode = 0; episode < episodes; episode++) 
    {
        Console.WriteLine("-----Starting episode"+episode+"---");
        int currentRow=11 ; int currentCol= 5;
        
        while (!hasHitWallOrEndOfMaze(currentRow, currentCol, floorValue))
        {
            long currentAction= determinedenxtAction(currentRow, currentCol, epsilon);
            int previousRow = currentRow;
            int previousCol= currentCol;
            (int, int) nextMove = moveOneSpace(maze,currentRow, currentCol, currentAction);

            currentRow=nextMove.Item1 ; currentCol=nextMove.Item2 ;

            float reward = rewards[currentRow, currentCol];
            float previousQvlaue = Qvalues[previousRow, previousCol, currentAction].item<float>();
            float temporalDifference = reward + (discountFactor * torch.max(Qvalues[currentRow, currentCol])).item<float>()- previousQvlaue ;
            float nextQvalue = previousQvlaue + (learningRate * temporalDifference);
            Qvalues[previousRow, previousCol, currentAction]= nextQvalue;
        }
        Console.WriteLine("-----Starting episode" + episodes + "---");
    }
    Console.WriteLine("Complete training!");
}
// allow the AI to navigate inside the maze
List<int[]> navigateMaze(int[,] maze, int startRow, int startCol,int floorValue, int wallValue)
{
    List<int[]> path = new List<int[]>();
    if (hasHitWallOrEndOfMaze(startRow, startCol, floorValue))
    {
        return [];
    }
    else 
    {
        int currentRow= startRow;
        int currentCol= startCol;
        path = [[currentRow,currentCol]];

        while (!hasHitWallOrEndOfMaze(currentRow,currentCol, floorValue))
        {
            int nextAction= (int) (determinedenxtAction(currentRow,currentCol,1.0f));
            (int, int) nextMove = moveOneSpace(maze, currentRow, currentCol, nextAction);
            
            currentCol = nextMove.Item2;
            currentRow = nextMove.Item1;

            if (rewards[currentRow, currentCol] != wallValue)
            {
                path.Add([currentRow,currentCol]);
            }
            else {  continue;   }
        }
    }
    int MoveCount = 1;
    for (int i = 0; i < path.Count; i++)
    {
        Console.Write("Move" + MoveCount+" : (");
        foreach (int element in path[i]) 
        {
            Console.Write(" "+element);
        }
        Console.Write(" )");
        Console.WriteLine();
        MoveCount++;

    }
    return path;
}

const float EPSILON = 0.95f;
const float DICOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const float EPISODES = 1500;
const int Start_row = 11;
const int Start_col = 5;

setupRewards(maze1,wall_reward_value,floor_reward_value,goal_reward_value);
setupQvalues(maze1);
trainModel(maze1, floor_reward_value, EPSILON,DICOUNT_FACTOR, LEARNING_RATE, EPISODES);
navigateMaze(maze1,Start_row,Start_col,floor_reward_value,wall_reward_value);
