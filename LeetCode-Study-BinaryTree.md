# 前言

- 定义一个二叉树

```java
public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode next;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
    
    TreeNode(int _val, TreeNode _left, TreeNode _right, TreeNode _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}
```

# 一、二叉树遍历

## 1. 前序遍历

前序遍历首先访问根节点，然后遍历左子树，最后遍历右子树。

```java
public class PreorderTraversal {
    private List<Integer> ans;

    public List<Integer> preorderTraversal(TreeNode root) {
        ans = new ArrayList<>();
        dfs(root);
        return ans;
    }

    /*
     * 递归实现二叉树前序遍历。
     * 递归终止条件：当前根为空，返回空；
     */
    public void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        ans.add(root.val);
        dfs(root.left);
        dfs(root.right);
    }

    /**
     * 前序遍历非递归实现.
     */
    public List<Integer> preorderTraversal1(TreeNode root) {
        ans = new ArrayList<>();
        // 根节点为空，直接返回
        if (root == null) {
            return ans;
        }
        // 创建栈
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        //根节点复制一份传给node, node表示当前节点
        TreeNode node = root;
        // 当栈不为空或者node不为空时，继续循环
        while (!stack.isEmpty() || node != null) {
            // 当前节点不为空时，继续从左子树向下循环
            while (node != null){
                ans.add(node.val);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        return ans;
    }
}
```

- Deque.pop()：从此双端队列所表示的堆栈中弹出一个元素。
- Deque.push()：将一个元素推入此双端队列表示的栈中，即从此双端队列的头部添加元素。

## 2. 中序遍历

中序遍历是先遍历左子树，然后访问根节点，然后遍历右子树。

```java
public class InorderTraversal {
    private List<Integer> ans;

    public List<Integer> inorderTraversal(TreeNode root) {
        ans = new ArrayList<>();
        dfs(root);
        return ans;
    }

    public void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        ans.add(root.val);
        dfs(root.right);
    }

    public List<Integer> inorderTraversal1(TreeNode root) {
        ans = new ArrayList<>();
        // 空树判断
        if (root == null) {
            return ans;
        }

        // 利用双端队列创建一个栈
        Deque<TreeNode> stack = new LinkedList<TreeNode>();

        TreeNode node = root;

        // 栈不为空或节点不为空则继续循环。
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                // 节点入栈
                stack.push(node);
                // 向左走，当没有左子树时跳出循环。
                node = node.left;
            }
            node = stack.pop();
            ans.add(node.val);
            node = node.right;
        }
        return ans;
    }
}
```

## 3. 后序遍历

后序遍历是先遍历左子树，然后遍历右子树，最后访问树的根节点。

```java
public class PostorderTraversal {
    private List<Integer> ans;

    public List<Integer> postorderTraversal(TreeNode root) {
        ans = new ArrayList<>();
        dfs(root);
        return ans;
    }

    public void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        dfs(root.right);
        ans.add(root.val);
    }

    /**
     * 后序遍历非递归实现.
     */
    public List<Integer> postorderTraversal1(TreeNode root) {
        /*
         * 1. 每拿到一个节点，就把它保存在栈中
         * 2. 继续对这个节点的左子树重复过程1，直到左子树为空
         * 3. 因为保存在栈中的节点都遍历了左子树但是没有遍历右子树，所以对栈中节点出栈，并对它的右子树重复过程1直到遍历完所有节点
         */
        ans = new ArrayList<>();
        // 根节点为空，直接返回
        if (root == null) {
            return ans;
        }
        // 创建栈
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        //根节点复制一份传给node, node表示当前节点
        TreeNode node = root;
        // 当栈不为空或者node不为空时，继续循环
        while (!stack.isEmpty() || root != null) {
            // 当前节点不为空时，继续从左子树向下循环
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            // 将最左边的节点取出来，然后判断它有没有右子树，以及
            root = stack.pop();
            if (root.right == null || root.right == node) {
                ans.add(root.val);
                node = root;
                root = null;
            } else {
                stack.push(root);
                root = root.right;
            }
        }
        return ans;
    }
}
```

## 4. 层序遍历

即逐层地，从左到右访问所有节点。

```java
public class LevelOrderTraversal {
    private List<List<Integer>> ans;

    /**
     * 递归思路，BFS
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        ans = new ArrayList<>();
        dfs(ans, root, 0);
        return ans;
    }

    /*
     * 递归的思路，核心是在每次往下走一层时，往list中加入一个初始化的列表，然后在遍历到哪一层时，将它对应的值加入到它对应的层中。
     */
    public void dfs(List<List<Integer>> ans, TreeNode root, int level) {
        // 边界条件判断
        if (root == null) {
            return;
        }
        // level代表层数，如果 level >= list.size()，说明到下一层了，所以要先把下一层的list初始化，防止下面的add的时候空指针异常。
        if (level >= ans.size()) {
            ans.add(new ArrayList<>());
        }
        // level并表示的是第几层，这里访问到第几层，就把数据加入到第几层
        ans.get(level).add(root.val);
        // 当前节点访问完之后，在使用递归的方式分别访问当前节点的左右子节点。
        dfs(ans, root.left, level + 1);
        dfs(ans, root.right, level + 1);
    }

    /**
     * 层级遍历非递归思路，利用队列。
     */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        ans = new ArrayList<List<Integer>>();

        // 根节点为空直接返回
        if (root == null) {
            return ans;
        }
        /*
         * 思路：根节点先入队。然后循环1-3直到队列为空。
         * 1. 节点出队。
         * 2. 节点的左子树入队。
         * 3. 节点的右子树入队。
         */
        // 创建一个队列进行存储节点数据。
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        // 根节点入队列
        queue.add(root);
        while (!queue.isEmpty()) {
            // 此列表存储每一层的数据
            ArrayList<Integer> level = new ArrayList<>();
            // 每一层的数据个数
            int levelCount = queue.size();

            for (int i = 0; i < levelCount; i++) {
                // 节点出队
                TreeNode node = queue.remove();
                // 左节点入队
                if (node.left != null) {
                    queue.add(node.left);
                }
                // 右节点入队
                if (node.right != null) {
                    queue.add(node.right);
                }
                level.add(node.val);
            }
            ans.add(level);
        }
        return ans;
    }
}
```

# 二、运用递归解决问题

## 1. 二叉树的最大深度

```java
public class MaxDepth {
    /*
     * 1. DFS：深度优先搜索算法。对于树而言，就相当于沿着树的深度遍历树的节点，尽可能深的搜索树的分支，直到所有节点遍历完，
     * 2. BFS：广度优先搜索算法。对于树而言，相当于层序遍历，有多少层，则最大深度就是几。
     */

    /**
     * DFS递归实现，
     */
    public int maxDepthDFSRecursion(TreeNode root) {
        /*
         * 每个节点的深度与它的左右子树的深度有关，等于其左右子树最大深度加1；
         * 那么将求根节点的最大深度划分为求它的的左右子树的最大深度，以此类推；
         * 当当前节点为空时，那么递归终止，返回0；
         */
        if (root == null) {
            return 0;
        } else {
            // 获取左子树的深度
            int leftHeight = maxDepthDFSRecursion(root.left);
            // 获取右子树的深度
            int rightHeight = maxDepthDFSRecursion(root.right);
            // 返回其左右子树最大深度加1
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }

    /*
     * 当树非常深时，栈会发生由于保存过多的临时变量而溢出。事实上，函数调用的参数是通过栈空间来传递的，在调用过程中会占用线程的栈资源。
     * 而递归调用，只有走到最后的结束点后函数才能依次退出，而未到达最后的结束点之前，占用的栈空间一直没有释放，
     * 如果递归调用次数过多，就可能导致占用的栈资源超过线程的最大值，从而导致栈溢出，导致程序的异常退出。
     * 99%的递归转非递归，都可以通过栈来进行实现。
     */

    /**
     * DFS非递归实现，利用栈.
     */
    public int maxDepthDFSNonRecursion(TreeNode root) {
        /*
         * 利用深度优先算法遍历二叉树，同时求得最大深度。
         * 深度优先遍历又分为：前序、中序、后序；
         * 此处用深度前序遍历二叉树，添加一个层数栈，来记录每个节点所处的位置
         * 层数栈的值与节点栈一一对应，同入同出
         */
        // 记录二叉树的最大深度
        int max = 0;
        if (root == null) {
            return max;
        }
        // 节点栈
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        // 节点层数栈
        Stack<Integer> levelStack = new Stack<Integer>();
        nodeStack.push(root);
        levelStack.push(1);
        while (!nodeStack.empty()) {
            // 弹出节点与其对应的层数
            TreeNode curNode = nodeStack.pop();
            int curNodelevel = levelStack.pop();
            // 记录最大层数
            max = Math.max(max, curNodelevel);
            // 因为是前序遍历（根-左-右），所以右子树先入栈。
            if (curNode.right != null) {
                nodeStack.push(curNode.right);
                levelStack.push(curNodelevel + 1);
            }
            if (curNode.left != null) {
                nodeStack.push(curNode.left);
                levelStack.push(curNodelevel + 1);
            }
        }
        return max;
    }

    /**
     * BFS递归实现
     */
    public int maxDepthBFS(TreeNode root){
        /*
         * 二叉树层序遍历的递归法，最后返回遍历列表的size即可。
         */
        List<List<Integer>> ans = new ArrayList<>();
        dfs(ans, root, 0);
        return ans.size();
    }

    public void dfs(List<List<Integer>> ans, TreeNode root, int level) {
        // 边界条件判断
        if (root == null) {
            return;
        }
        // level代表层数，如果 level >= list.size()，说明到下一层了，所以要先把下一层的list初始化，防止下面的add的时候空指针异常。
        if (level >= ans.size()) {
            ans.add(new ArrayList<>());
        }
        // level并表示的是第几层，这里访问到第几层，就把数据加入到第几层
        ans.get(level).add(root.val);
        // 当前节点访问完之后，在使用递归的方式分别访问当前节点的左右子节点。
        dfs(ans, root.left, level + 1);
        dfs(ans, root.right, level + 1);
    }

    /**
     * BFS非递归实现，利用队列.
     */
    public int maxDepthBFSNonRecursion(TreeNode root) {
        /*
         * 层序遍历：根节点先入队，然后循环1~3至队列为空
         * 1. 节点出队
         * 2. 节点的左子树入队
         * 3. 节点的右子树入队
         */
        if (root == null) {
            return 0;
        }
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        stack.add(root);
        TreeNode outNode = root;
        // count 记录二叉树的深度
        int count = 1;
        while (!stack.isEmpty()) {
            // 记录每一层节点的个数；
            int levelCount = stack.size();
            // outNode为栈队列出队的节点
            for (int i = 0; i < levelCount; i++) {
                outNode = stack.pop();
                if (outNode.left != null) {
                    stack.add(outNode.left);
                }
                if (outNode.right != null) {
                    stack.add(outNode.right);
                }
            }
            count++;
        }
        return count;
    }
}
```

## 2. 对称二叉树

```java
public class IsSymmetric {

    /**
     * 递归法
     */
    public boolean isSymmetric3(TreeNode root) {
        /*
         * 将两个树的对称问题理解为：它们的根节点具有相同的值，且每个树的右子树都与另一个树的左子树镜像对称。
         * 方法：定义两个指针，p指针与q指针，两个指针初始化指向树的根节点，
         * 然后p右移时，q左移；p左移时，q右移；
         * 每次移动时检查指针指向的节点的值是否相等，若相等，再向下移。
         */
        return dfs3(root, root);
    }

    public boolean dfs3(TreeNode p,TreeNode q) {
        if(p == null && q == null){
            return true;
        }
        if(p == null || q == null){
            return false;
        }
        return p.val == q.val && dfs3(p.left,q.right) && dfs3(p.right,q.left);
    }

    /**
     * 非递归法
     */
    public boolean isSymmetric4(TreeNode root){
        /*
         * 创建一个队列，将左右子树放入队列中，每次出队两个节点，只要每次都一样，那么就是true
         * 初始化队列时根节点入队两次，当队列为空时，树遍历完了
         */
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        // 根节点入队两次
        nodeQueue.add(root);
        nodeQueue.add(root);
        while (!nodeQueue.isEmpty()){
            TreeNode leftNode = nodeQueue.remove();
            TreeNode rightNode = nodeQueue.remove();
            if(leftNode == null && rightNode == null){
                continue;
            }
            if((leftNode == null || rightNode == null) || leftNode.val != rightNode.val){
                return false;
            }
            nodeQueue.add(leftNode.left);
            nodeQueue.add(rightNode.right);
            nodeQueue.add(leftNode.right);
            nodeQueue.add(rightNode.left);
        }
        return true;
    }


    /*      自己写的      */

    /**
     * 递归法
     */
    public boolean isSymmetric1(TreeNode root) {
        /*
         * 一个列表存储根节点的左子树，存储顺序：根-左-右
         * 一个列表存储根节点的右子树，存储顺序：根-右-左
         * 并且当当前节点没有左右子树时，用一个符号“101”进行标记存储到列表中去，相当于“占位置”
         */
        // 边界判断
        if (root == null) {
            return false;
        }
        // 存储根节点的左子树列表
        ArrayList<Integer> leftSubtree = new ArrayList<Integer>();
        // 存储根节点的左子树
        ArrayList<Integer> rightSubtree = new ArrayList<Integer>();
        dfs1(root.left, leftSubtree);
        dfs2(root.right, rightSubtree);
        // 若两个列表长度不同，则直接返回false
        if (leftSubtree.size() != rightSubtree.size()) {
            return false;
        } else {
            // 若相同，则循环遍历，当有不同的值时，返回false，完全相同则返回true
            for (int i = 0; i < leftSubtree.size(); i++) {
                if (leftSubtree.get(i) != rightSubtree.get(i)) {
                    return false;
                }
            }
            return true;
        }
    }

    public void dfs1(TreeNode root, List<Integer> leftSubTree) {
        if (root == null) {
            leftSubTree.add(0);
            return;
        } else {
            leftSubTree.add(root.val);
            dfs1(root.left, leftSubTree);
            dfs1(root.right, leftSubTree);
        }
    }

    public void dfs2(TreeNode root, List<Integer> rightSubTree) {
        if (root == null) {
            rightSubTree.add(0);
            return;
        } else {
            rightSubTree.add(root.val);
            dfs2(root.right, rightSubTree);
            dfs2(root.left, rightSubTree);
        }
    }

    /**
     * 非递归法
     */
    public boolean isSymmetric2(TreeNode root) {
        /*
         * 从根节点将儿叉树分为左子树与右子树，然后对这两个子树进行遍历，若果每次遍历的值都相同，那么最终返回true，否则返回false
         * 左子树遍历规则：根-左-右；右子树遍历规则：根-右-左
         * 用两个队列来存储每次遍历的值
         */
        // 边界判断
        if (root == null) {
            return false;
        }
        // 将根节点分为左右子树
        TreeNode leftSubtreeNode = root.left;
        TreeNode rightSubtreeNode = root.right;
        // 当左右子树都为空时，返回true
        if (leftSubtreeNode == null && rightSubtreeNode == null) {
            return true;
        }

        // 创建左右子树队列
        Queue<TreeNode> leftSubtreeNodeQueue = new LinkedList<TreeNode>();
        Queue<TreeNode> rightSubtreeNodeQueue = new LinkedList<TreeNode>();

        leftSubtreeNodeQueue.add(leftSubtreeNode);
        rightSubtreeNodeQueue.add(rightSubtreeNode);

        // 当其中一个子树为空，另一个不为空时，返回false
        if ((leftSubtreeNode == null && rightSubtreeNode != null) ||
                (leftSubtreeNode != null && rightSubtreeNode == null)) {
            return false;
        }

        while (!leftSubtreeNodeQueue.isEmpty() && !rightSubtreeNodeQueue.isEmpty()) {

            leftSubtreeNode = leftSubtreeNodeQueue.remove();
            rightSubtreeNode = rightSubtreeNodeQueue.remove();
            // 节点出队，判断节点值是否相等，若不相等，则直接返回false
            if (leftSubtreeNode.val == rightSubtreeNode.val) {
                // 按照遍历顺序往队列中添加节点，当其中一个节点为空，另一个不为空时，返回false
                if (leftSubtreeNode.left != null && rightSubtreeNode.right != null) {
                    leftSubtreeNodeQueue.add(leftSubtreeNode.left);
                    rightSubtreeNodeQueue.add(rightSubtreeNode.right);
                } else if ((leftSubtreeNode.left != null && rightSubtreeNode.right == null) ||
                        (leftSubtreeNode.left == null && rightSubtreeNode.right != null)) {
                    return false;
                }
                if (leftSubtreeNode.right != null && rightSubtreeNode.left != null) {
                    leftSubtreeNodeQueue.add(leftSubtreeNode.right);
                    rightSubtreeNodeQueue.add(rightSubtreeNode.left);
                } else if ((leftSubtreeNode.right != null && rightSubtreeNode.left == null) ||
                        (leftSubtreeNode.right == null && rightSubtreeNode.left != null)) {
                    return false;
                }
            } else {
                return false;
            }
            // 当其中一个队列为空，另一个不为空时，返回false
            if ((!leftSubtreeNodeQueue.isEmpty() && rightSubtreeNodeQueue.isEmpty()) ||
                    (leftSubtreeNodeQueue.isEmpty() && !rightSubtreeNodeQueue.isEmpty())) {
                return false;
            }
        }
        return true;


    }
}
```

## 3. 路径总和

```java
public class HasPathSum {
    /**
     * 112. 路径总和
     */
    /**
     * 递归法
     */
    public boolean hasPathSum1(TreeNode root, int targetSum) {
        /*
         * 遍历二叉树，定义一个curTarget表示targetSum减去路径上每个节点的值后剩余的值，curTarget初始化为targetSum
         * 子问题为，当到root.right或root.left时，targetSum = targetSum - root.val;
         * 这样依次往下，当节点为null时，返回false，当递归到某个叶子节点(左右子树都为null)时且当前节点的val == targetSum时，返回true。
         * 每次的返回结果进行或运算，有一个true则说明有存在值相加等于targetSum的路径。
         */
        // 当前节点为null时，返回false
        if (root == null) {
            return false;
        }
        // 判断当前节点是否是叶子节点，若是则判断值是否相等
        if (root.left == null && root.right == null) {
            return root.val == targetSum;
        }
        return hasPathSum1(root.left, targetSum - root.val) || hasPathSum1(root.right, targetSum - root.val);
    }

    /**
     * 迭代法
     */
    public boolean hasPathSum2(TreeNode root, int targetSum) {
        /*
         * 利用两个队列，一个队列存当前节点，一个队列存根节点到当前节点值的和，用BFS遍历法进行遍历求解
         * 每次节点值和节点一块出队列时，然后进行一个判断，如果当前节点为叶子节点，那么就比对一下targetSum与当前路径上的和是否相等，然后返回
         */
        // 边界处理
        if(root == null){
            return false;
        }
        // 定义两个队列
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        Queue<Integer> pathSumQueue = new LinkedList<Integer>();
        /*
         * 同样是往列表结尾添加元素，LinkedList中add()与offer()的区别：
         * offer 方法 调用的就是 add 方法
         * offer 实现 Deque 接口的方法，add 实现 Collection 接口方法
         * 作为List使用时,一般采用add / get方法来 加入/获取对象
         * 作为Queue使用时,才会采用 offer/poll/take等方法
         */
        nodeQueue.add(root);
        pathSumQueue.offer(root.val);
        while (!nodeQueue.isEmpty()){
            TreeNode curNode = nodeQueue.poll();
            int curPathSum = pathSumQueue.poll();
            // 判断是不是叶子节点，如果是，则进行比对
            if(curNode.left == null && curNode.right == null){
                if(curPathSum == targetSum){
                    return true;
                }
            }
            if(curNode.left!=null){
                nodeQueue.offer(curNode.left);
                pathSumQueue.offer(curPathSum+curNode.left.val);
            }
            if(curNode.right!=null){
                nodeQueue.offer(curNode.right);
                pathSumQueue.offer(curPathSum+curNode.right.val);
            }
        }
        return false;
    }
}
```

# 三、总结

## 1. 从前序与中序遍历序列构造二叉树

```java
public class PreorderAndInorderBuildTree {
    /**
     * 105. 从前序与中序遍历序列构造二叉树
     */
    // 创建成员变量，方便调用
    private int post_idx;
    private int[] preorder;
    private int[] inorder;
    private Map<Integer, Integer> idx_map = new HashMap<Integer, Integer>();

    /**
     * 递归法
     */
    public TreeNode preorderAndInorderBuildTreeq1(int[] preorder, int[] inorder) {
        /*
         * 前序遍历：根 -> 左 -> 右
         * 中序遍历：左 -> 根 -> 右
         * 与 106 题类似的思想
         * 前序遍历数组的第一个元素便是二叉树的根节点，所以根据根节点在中序遍历数组中的位置，可以将中序数组分为左右子树
         * 算法思路：首先创建一个哈希表存储中序遍历数组的元素与下标 <数组元素，数组下标>
         *         定义一个递归函数 helper(in_left, in_right) 表示当前子树的边界，in_right 初始化为 inorder.length - 1
         *         如果 in_left > in_right 表示当前子树为空，返回空；
         *         然后根据哈希表返回前序遍历数组中当前根节点元素在中序遍历数组中的位置下标 index ；
         *         根据 index 得到左子树 (in_left, index-1) ，右子树 (index+1, in_right)
         *         然后再继续递归遍历，此处要先生成左子树，在生成右子树，因为前序遍历是：根 -> 左 -> 右
         *
         */
        // 初始化
        this.preorder = preorder;
        this.inorder = inorder;

        post_idx = 0;
        // 将中序遍历数组元素存入 idx_map
        int idx = 0;
        for (Integer val : inorder) {
            idx_map.put(val, idx++);
        }
        return helper(0, inorder.length - 1);
    }

    private TreeNode helper(int in_left, int in_right) {
        // 如果 in_left > in_right 说明当前子树为空，返回空
        if (in_left > in_right) {
            return null;
        }

        // 获取根结点的值，并建立根节点
        int root_val = preorder[post_idx];
        TreeNode root = new TreeNode(root_val);
        // 根节点向右移动一位
        post_idx++;

        // 获取根节点在中序遍历数组中的 index
        int index = idx_map.get(root_val);
        // 递归建立左右子树，先创建左子树，在创建右子树
        root.left = helper(in_left, index - 1);
        root.right = helper(index + 1, in_right);

        return root;
    }

    /**
     * 迭代法
     */
    public TreeNode preorderAndInorderBuildTreeq2(int[] preorder, int[] inorder) {
        /*
         * 用一个栈来辅助完成树的建立。
         * 思路：前序遍历数组中任意两个节点 u 和 v，他们之间的关系只有两种：
         * 1. v 是 u 的左儿子；
         * 2. u 没有左儿子，那么 v 将是 u 的祖先节点右儿子或者是 u 的左儿子，如果 u 没有右儿子，那么就向上回溯，
         * 因为前序遍历是：根 -> 左 -> 右，中序遍历是：左 -> 根 -> 右，
         * 那么将前序遍历数组入栈，入栈时判断栈顶元素与中序遍历数组当前值是否相同，
         * 若不相同，则证明当前节点还有左儿子，则当前前序遍历数组节点入栈继续判断
         * 若相同，则证明当前前序遍历数组的节点已经没有左儿子了，下一个前序遍历数组元素不是当前节点的右儿子，就是其祖先节点的右儿子
         * 那么然后开始出栈，出栈条件：当栈不为空且栈顶元素节点值 == 中序遍历数组当前值时，出栈；每出一个中序遍历数组元素向右移一位；
         * 出栈是为了找到前序遍历数组的下一个元素的父节点。
         * 出完后，将前序遍历数组的下一个元素，作为当前节点的右儿子，然后入栈。
         * 循环遍历。
         */
        // 边界判断
        if (preorder.length == 0 || preorder == null) {
            return null;
        }
        // 因为前序遍历数组第一个节点为根节点，所以先创建树的根节点
        TreeNode root = new TreeNode(preorder[0]);
        // 创建一个栈
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        // 根节点入栈
        stack.push(root);
        // 创建一个变量表示中序数组的下标
        int inorderIndex = 0;
        for (int i = 1; i < preorder.length; i++) {
            // 获取前序遍历数组当前的值
            int preorderVal = preorder[i];
            // 获取当前节点
            TreeNode node = stack.peek();
            // 判断当前节点的值与当前中序遍历数组的值是否相等
            if (node.val != inorder[inorderIndex]) {
                // 若不相等，则表明当前的 preorderVal 是当前节点的左儿子，那么将其创建并入栈
                node.left = new TreeNode(preorderVal);
                stack.push(root.left);
            } else {
                // 若不想等，则证明当前的 preorderVal 不是当前节点的左儿子了，那么 preorderVal 要么是当前节点的右儿子，要么是当前节点祖先的右儿子
                // 寻找 preorderVal 的父节点
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]) {
                    // 向上回溯当前节点的祖先节点
                    node = stack.pop();
                    inorderIndex++;
                }
                node.right = new TreeNode(preorderVal);
                stack.push(node.right);
            }
        }
        return root;
    }

    /**
     * 递归法
     */
    public TreeNode preorderAndInorderBuildTreeq3(int[] preorder, int[] inorder) {
        /*
         * 思路大致与 preorderAndInorderBuildTreeq1 一致。
         * 前序遍历：[根节点，[左子树的前序遍历结果]，[右子树的前序遍历结果]]
         * 中序遍历：[[左子树的中序遍历结果]，根节点，[右子树的中序遍历结果]]
         * 核心思想：左子树中序遍历的节点数等于前序遍历的节点数。
         * 定义一个集合，存储中序遍历数组 <数组元素，数组下标>，
         * 定义一个函数 myBuildTree 传入参数：前序数组，中序数组，前序数组开始位置，前序数组结束位置，中序数组开始位置，中序数组结束位置
         * 这样可以将左右子树当做一个完整的树传入 myBuildTree 中，递归建立树
         */
        // 获取数组长度
        int n = preorder.length - 1;
        // 定义下标
        int index = 0;
        // 中序数组存入集合
        for (Integer val : inorder) {
            idx_map.put(val, index++);
        }
        return myBuildTree(preorder, inorder, 0, n, 0, n);
    }

    private TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }
        // 前序遍历数组的第一个节点就是根节点
        int preorder_root_val = preorder[preorder_left];
        // 创建根节点
        TreeNode root = new TreeNode(preorder_root_val);
        // 获取根节点在中序数组中的位置
        int inorder_root_index = idx_map.get(preorder_root_val);
        // 得到左子树的节点的数目
        int size_left_subtree = inorder_root_index - inorder_left;
        // 前序遍历左子树位置：[preorder_left+1, preorder_left+size_left_subtree] 对应中序中的左子树位置[inorder_left, inorder_root_index-1]
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root_index - 1);
        // 前序遍历右子树位置：[preorder_left+size_left_subtree+1, preorder_right 对应中序中的右子树位置[inorder_root_index+1, inorder_right]
        root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root_index + 1, inorder_right);
        return root;
    }
}
```

## 2. 从中序与后序遍历序列构造二叉树

```java
public class InorderAndPostorderBuildTree {
    /**
     * 106. 从中序与后序遍历序列构造二叉树
     */
    // 创建一些成员变量，方便调用
    private int post_idx; // 表示树的根节点
    private int[] inorder; // 中序遍历的数组
    private int[] postorder; // 后序遍历的数组
    private Map<Integer, Integer> idx_map = new HashMap<Integer, Integer>();

    /**
     * 递归法
     */
    public TreeNode inorderAndPostorderBuildTree1(int[] inorder, int[] postorder) {
        /*
         * 树的中序遍历：左 -> 根 -> 右
         * 树的后序遍历：左 -> 右 -> 根
         * 规律：后序数组的最后一个元素即为根节点。那就可以根据后续数组的最后一个元素将中序数组分成[左子树数组]，[根节点]，[右子树数组]
         * 然后再将左子树数组，与右子树数组再按照上面的方法继续递归的划分下去，直到最终出现了叶节点。
         * 算法过程：首先定义一个哈希表来存储 <数组元素，数组下标>
         *         定义一个递归函数 helper(in_left, in_right) 表示当前递归到中序序列中当前子树的边界，递归入口为 helper(0, n-1)
         *         如果 in_left > in_right 说明子树为空，返回空节点；
         *         选择后序遍历的最后一个节点作为根节点；
         *         然后通过哈希表查到根节点 index ，从 in_left 到 index-1 为左子树，从 index+1 到 in_right 为右子树；
         *         根据后序遍历的规律：递归创建右子树 helper(index+1, in_right)，然后再递归创建左子树 helper(in_left, index-1);
         *         因为后序是左右根，所以逆向创建时也应该是先创建右子树，在创建左子树。
         */
        // 成员属性初始化
        this.inorder = inorder;
        this.postorder = postorder;
        // 从后序遍历数组的最后一个元素开始
        post_idx = postorder.length - 1;
        // 定义一个idx表示中序数组的下标
        int idx = 0;
        // 将中序数组元素存入哈希表中
        for (Integer val : inorder) {
            idx_map.put(val, idx++);
        }
        return helper(0, inorder.length - 1);
    }

    private TreeNode helper(int in_left, int in_right) {
        // in_left > in_right 说明子树为空，返回空
        if (in_left > in_right) {
            return null;
        }

        // 选择后序数组中 post_idx 的位置的元素作为当前子树的根节点
        int root_val = postorder[post_idx];
        TreeNode root = new TreeNode(root_val);

        // 根据中序数组中 root 值所在的位置将数组分成两棵左右子树
        int index = idx_map.get(root_val);

        // 根节点下标减一
        post_idx--;

        // 构建右子树
        root.right = helper(index + 1, in_right);

        // 构建左子树
        root.left = helper(in_left, index - 1);

        return root;
    }

    /**
     * 迭代法
     */
    public TreeNode inorderAndPostorderBuildTree2(int[] inorder, int[] postorder) {
        /*
         * 用一个栈来辅助完成树的建立。
         * 后序遍历是：[左 -> 右 -> 根]
         * 中序遍历是：[左 -> 根 -> 右]
         * 对后序遍历数组从右往左入栈，那么入栈的节点将是：二叉树的根节点，根节点的右儿子，右儿子的右儿子，依次往下，[根->右儿子->右儿子……]。方向是从根向叶的
         * 而中序遍历从右往左看的话，那么将是：右节点，右节点的父节点，父节点的父节点，一直到根节点，依次往上，[右儿子->父节点……->根节点]。方向是从叶到根
         * 后序数组是从右向左开始入栈的，那么当入栈到二叉树的最右边的右子节点时，此时出栈的顺序将与中序数组从右往左的顺序是相同的
         * 遍历后序数组，从右边第二个元素开始往左遍历，入栈时进行判断，判断条件：当前栈顶节点的值是否等于当前中序遍历数组的值
         * 若不等于：当前后续数组元素为当前节点的右儿子，然后入栈；
         * 若等于：则证明当前节点没有右儿子了，那么就要找出当前前序数组元素是属于当前节点的左儿子还是属于当前节点他祖先的左儿子；
         */
        // 边界处理
        if (postorder.length == 0 || postorder == null) {
            return null;
        }
        // 获取根节点
        TreeNode root = new TreeNode(postorder[postorder.length - 1]);
        // 创建一个栈
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        // 根节点入栈
        stack.push(root);
        int inorderIndex = inorder.length - 1;
        for (int i = postorder.length - 2; i >= 0; i--) {
            // 获取当前后序数组遍历的值
            int postorderVal = postorder[i];
            // 获取当前头结点
            TreeNode node = stack.peek();
            if (node.val != inorder[inorderIndex]) {
                node.right = new TreeNode(postorderVal);
                stack.push(node.right);
            } else {
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]) {
                    node = stack.pop();
                    inorderIndex--;
                }
                node.left = new TreeNode(postorderVal);
                stack.push(node.left);
            }
        }
        return root;
    }
}
```

## 3. 填充每个节点的下一个右侧节点指针

```java
public class ConnectRightNode {
    /**
     * 116. 填充每个节点的下一个右侧节点指针
     */
    /**
     * 层序遍历、队列
     */
    public TreeNode connect1(TreeNode root) {
        /*
         * 层序遍历
         * 利用队列，题目是要求二叉树的每一层的节点都连接起来形成一个链表。
         * 每次出队节点的 next 指向队首的节点。
         */
        if (root == null) {
            return null;
        }
        // 创建一个队列
        Deque<TreeNode> deque = new LinkedList<TreeNode>();
        // 根节点入队
        deque.add(root);
        while (!deque.isEmpty()) {
            // 得到每一层节点的个数
            int size = deque.size();
            // 一层出队，每个节点左右子节点入队。
            for (int i = 0; i < size; i++) {
                // 从队首取出元素
                TreeNode node = deque.remove();
                // 将每次出对的节点都连接起来
                if (i < size - 1) {
                    node.next = deque.peek();
                }
                // 出队节点的左右子节点入队
                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
                }
            }
        }
        return root;
    }

    /**
     * 层序遍历、原地操作
     */
    public TreeNode connect2(TreeNode root) {
        /*
         * 原地操作
         * 使用已有的 next 指针，假设二叉树有 N 层，每次的 N-1 层，处理第 N 层的节点
         * 你想想：下一层的节点无非就两种关系，一是同一个父节点，二是不同父节点；
         * 对于同一个父节点：node.left.next = node.right;
         * 对于不同一个父节点的两个节点，若他们两个有关系，那么它们的父节点一定是：一个是一个的下一个节点
         * 那么就会有：node.right.next = node.next.left
         * 每一层按照上述方法，一直到倒数第二层。
         */
        // 边界处理
        if (root == null) {
            return root;
        }
        // 从根节点开始，一层一层往下处理，leftmost作为第N-1层节点的头节点，再一个就是用来判断是否到达最后一层
        TreeNode leftmost = root;
        // 当到最后一层时，终止循环
        while (leftmost.left != null) {
            // 头结点复制一份，免得后面头节点的改变导致链表发生变化。
            TreeNode head = leftmost;
            while (head != null) {
                // 同一个父节点
                head.left.next = head.right;
                // 不同父节点
                if (head.next != null) {
                    head.right.next = head.next.left;
                }
                head = head.next;
            }
            leftmost = leftmost.left;
        }
        return root;
    }
    
    /**
     * 递归法
     */
    public TreeNode connect3(TreeNode root) {
        /*
         * 递归法
         * 思路：
         * 终止条件：当前节点为空时
         * 递归体：
         */
        dfs(root);
        return root;
    }

    private void dfs(TreeNode root) {
        if(root == null){
            return;
        }
        TreeNode nodeLeft = root.left;
        TreeNode nodeRight = root.right;
        while (nodeLeft!= null){

            nodeLeft.next = nodeRight;
            nodeLeft = nodeLeft.right;
            nodeRight = nodeRight.left;

        }
        dfs(root.left);
        dfs(root.right);
    }
}
```

## 4. 填充每个节点的下一个右侧节点指针 II

```java
public class ConnectRightNodeTwo {
    /**
     * 117. 填充每个节点的下一个右侧节点指针 II
     */
    /**
     * 层序遍历、队列
     */
    public TreeNode connect1(TreeNode root) {
        /*
         * 层序遍历
         * 利用队列，题目是要求二叉树的每一层的节点都连接起来形成一个链表。
         * 每次出队节点的 next 指向队首的节点。
         */
        if (root == null) {
            return null;
        }
        // 创建一个队列
        Deque<TreeNode> deque = new LinkedList<TreeNode>();
        // 根节点入队
        deque.add(root);
        while (!deque.isEmpty()) {
            // 得到每一层节点的个数
            int size = deque.size();
            // 一层出队，每个节点左右子节点入队。
            for (int i = 0; i < size; i++) {
                // 从队首取出元素
                TreeNode node = deque.remove();
                // 将每次出对的节点都连接起来
                if (i < size - 1) {
                    node.next = deque.peek();
                }
                // 出队节点的左右子节点入队
                if (node.left != null) {
                    deque.add(node.left);
                }
                if (node.right != null) {
                    deque.add(node.right);
                }
            }
        }
        return root;
    }

    /**
     * 层序遍历、原地操作
     */
    public TreeNode connect2(TreeNode root) {
        /*
         * 将每一层节点看做一个链表，然后给每一层的链表加上一个头节点，这样方便链接
         * 然后循环当前一层的链表节点，将当前层节点存在的左右子树依次连接到头节点上去
         */
        if(root == null){
            return root;
        }
        // 当前层的开始节点
        TreeNode curNode = root;
        while (curNode!=null){

            // 创建下一层的头节点
            TreeNode dummy = new TreeNode(0);
            // 下一层的头结点复制一份，防止头节点发生改动。
            TreeNode preNode = dummy;
            // 开始遍历当前层的节点
            while (curNode!=null){
                // 当前层节点的左子树不为空，那么接到下一层链表的后面
                if(curNode.left!=null){
                    preNode.next = curNode.left;
                    preNode = preNode.next;
                }
                // 当前层节点的右子树不为空，那么接到下一层链表的后面
                if(curNode.right != null){
                    preNode.next = curNode.right;
                    preNode = preNode.next;
                }
                // 当前层节点后移一位
                curNode = curNode.next;
            }
            // 将下一层的头节点的下一个节点作为当前层
            curNode = dummy.next;
        }
        return root;
    }
}
```

## 5. 二叉搜索树的最近公共祖先

```java
public class LowestCommonAncestor {
    /**
     * 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
     */
    /*
     * 什么是二叉搜索树？
     * 每个节点中的值必须大于（或等于）存储在其左子树中的任何值；
     * 每个节点中的值必须小于（或等于）存储在其右子树中的任何值；
     * 根节点的值大于等于其左子树中任意一个节点的值，小于等于其右子树中任意一节点的值。
     */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        /*
         * 两次遍历；
         * 从根节点开始查找 p、q，以查找 p 为例：
         * 若 root == p, 那直接返回就好；
         * 若 root ！= p， 则判断 root.val 与 p.val 的大小
         * 如果 root.val > p.val；说明 p 在 root 的左子树中
         * 如果 root.val < p.val；说明 p 在 root 的右子树中；
         * 将两个节点 p、q 的路径进行对比，找到最后的相同的节点就是公共祖先了；
         * 后面两个路径对比这块，感觉和 160. 相交链表 有点相似
         */
        // 边界处理
        if (root == null) {
            return root;
        }
        // 得到从根节点到两个目标节点的路径
        List<TreeNode> path_P = getPath(root, p);
        List<TreeNode> path_Q = getPath(root, q);
        // 定义祖先节点，初始化为根节点
        TreeNode ancestoNode = root;
        // 遍历路径，找到祖先节点
        for (int i = 0; i < path_P.size() && i < path_Q.size(); i++) {
            if (path_P.get(i) == path_Q.get(i)) {
                ancestoNode = path_P.get(i);
            } else {
                break;
            }
        }
        return ancestoNode;
    }

    private List<TreeNode> getPath(TreeNode root, TreeNode target) {
        List<TreeNode> path = new ArrayList<TreeNode>();
        // 根节点复制一份，防止根节点在后面的操作中发生改变
        TreeNode curNode = root;
        while (curNode != target) {
            // 列表先添加当前节点
            path.add(curNode);
            if (curNode.val > target.val) {
                curNode = curNode.left;
            } else {
                curNode = curNode.right;
            }
        }
        // 当前节点 == 目标节点时，加入列表
        path.add(curNode);
        return path;
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        /*
         * 一次遍历。
         * 从根部开始往下遍历。p 和 q 是不同的节点
         * 如果 curNode.val > p与q的值，那么说明 p、q 在当前节点的左子树中，那么 curNode = curNode.left
         * 如果 curNode.val < p与q的值，那么说明 p、q 在当前节点的右子树中，那么 curNode = curNode.right
         * 否则，p.val < curNode.val < q.val，则curNode为祖先节点
         * 不需要考虑 p 节点是 q 节点的祖先这种情况，上面的遍历会遍历到这种情况。
         */
        // 根节点复制一份
        TreeNode curNode = root;
        while (curNode != null) {
            if (curNode.val > p.val && curNode.val > q.val) {
                curNode = curNode.left;
            } else if (curNode.val < p.val && curNode.val < q.val) {
                curNode = curNode.right;
            } else {
                break;
            }
        }
        return curNode;
    }

    /**
     * 递归
     */
    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        /*
         * 与一次遍历思路一样，利用递归代替while循环
         */
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor3(root.left, p, q);
        }
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor3(root.right, p, q);
        }
        return root;
    }
}
```

## 6. 二叉树的最近公共祖先

```java
public class BinaryTreeLowestCommonAncestor {
    /**
     * 剑指 Offer 68 - II. 二叉树的最近公共祖先
     */
    /**
     * 迭代
     */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        /*
         * 用哈希表来存储 p、q节点的父节点，用BFS进行遍历树，只要遍历到 p、q时，就停止遍历
         * 然后在获取到根节点到 p 节点的路径，然后一层一层向上找 q 的父节点，只要第一个父节点出现在根节点到 p 节点的路径中，那么返回这个节点
         */
        if (root == null) {
            return root;
        }
        // 创捷一个哈希表来存储 <节点, 父节点>
        Map<TreeNode, TreeNode> parents = new HashMap<TreeNode, TreeNode>();
        // 根节点的父节点为 null
        parents.put(root, null);
        // 创建一个队列来 BFS
        Deque<TreeNode> deque = new LinkedList<TreeNode>();
        // 根节点入队
        deque.offer(root);
        // 此时入队时不需要再用队列为空来作为终止条件了，只要当 parents 中包含 p、q 节点时，就可以中止了
        while (!(parents.containsKey(p) && parents.containsKey(q))) {
            TreeNode curNode = deque.poll();
            if (curNode.left != null) {
                // 子节点与父结点存入哈希表中
                parents.put(curNode.left, curNode);
                // 子节点入队
                deque.offer(curNode.left);
            }
            if (curNode.right != null) {
                parents.put(curNode.right, curNode);
                deque.offer(curNode.right);
            }
        }
        // 创建一个集合来存储根节点到 p 节点的路径
        Set<TreeNode> path = new HashSet<TreeNode>();
        // 循环将根节点到 p 节点经过的所有节点存入集合中
        while (p != null) {
            // 从 p 节点开始，并包含 p 节点，万一 p 节点是 q 节点的祖先呢！
            path.add(p);
            p = parents.get(p);
        }
        // 如果 q 节点的父节点在path中没有出现，那么就在向上找父节点，如果在path中出现，那么返回这个出现的节点
        while (!path.contains(q)) {
            q = parents.get(q);
        }
        return q;
    }

    /**
     * 递归
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        /*
         * 
         */
        return null;
    }
}
```

## 7. 序列化与反序列化二叉树

```java
public class SeqToOppositeSeqBinaryTree {
    /**
     * 剑指 Offer II 048. 序列化与反序列化二叉树
     */

    /*
     * 先理解哈题
     *  Codec ser = new Codec();
     *  Codec deser = new Codec();
     *  TreeNode ans = deser.deserialize(ser.serialize(root));
     * 题目意思就是对输入的树先序列化成字符串，然后对字符串再进行解码成树，也就是说通过这两个过程，输入还要是输出。
     * 树的序列化：
     *      将树编码成字符串，用 "," 隔开每个节点，当前节点的子节点为空时用 "None" 来表示；
     *      那就是对数进行遍历了，遍历可以分为：DFS、BFS；其中 BFS 有可以理解为树的层序遍历，DFS 包含树的 先序、中序、后序三种遍历方法；
     *
     *
     */
    /* -----------------------------------------DFS：先序遍历 + 递归-----------------------------------------------*/
    // Encodes a tree to a single string.
    public String serialize1(TreeNode root) {
        /*
         * 终止条件：当前节点是叶子节点时，返回"None"
         * 否则当前节点的值 + "," + 再加上它的左子树的返回值 + "," + 它的右子树的返回值
         */
        if (root == null) {
            return "None";
        }
        return root.val + "," + serialize1(root.left) + "," + serialize1(root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize1(String data) {
        // 将字符串切割成字符数组
        String[] dataArray = data.split(",");
        for (int i = 0; i < dataArray.length; i++) {
            System.out.println(dataArray[i]);
        }
        // 将数组的元素放到队列里面去
        List<String> dataList = new LinkedList<>(Arrays.asList(dataArray));
        return buildTree(dataList);
    }

    private TreeNode buildTree(List<String> dataList) {
        /*
         * 递归建立树
         * 因为是前序遍历的数组，所以节点顺序为：根左右；
         * 当出队的元素为空时，递归返回，并移除队首元素
         * 否则就创建一个根节点，然按照先左后右的顺序将其接到递归上一层的节点上去
         */
        if (dataList.get(0).equals("None")) {
            dataList.remove(0);
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(dataList.get(0)));
        dataList.remove(0);
        root.left = buildTree(dataList);
        root.right = buildTree(dataList);
        return root;
    }

    /* -----------------------------------------BFS + 迭代-----------------------------------------------*/
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        /*
         * 层序遍历
         * 创建一个队列，当当前节点不为null时，它的左右子树入队，不考虑他的左右子树是否为空，然后给字符串加上当前结点的值
         * 若当前节点为null那么给字符串加上一个None；
         */
        if (root == null) {
            return "None";
        }
        String data = "";
        // 创建一个队列
        Deque<TreeNode> deque = new LinkedList<TreeNode>();
        // 根节点先入队
        deque.offer(root);
        while (!deque.isEmpty()) {
            // 获取队首元素
            TreeNode curNode = deque.poll();
            if(curNode!=null){
                data = data + Integer.valueOf(curNode.val).toString();
                deque.add(curNode.left);
                deque.add(curNode.right);
            }else {
                data = data + "None";
            }
            // 每次遍历一个节点都给后面加上一个','表间隔。
            data = data + ",";
        }
        return data;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize3(String data) {
        /*
         * 用队列来存储每个字符元素
         * 创建一个链表来存储每层的节点，当链表遍历完时，表示这一层的节点的子节点处理完了(此处应该用队列来处理可能会更好一点)
         */
        String[] dataArray = data.split(",");
        List<String> dataList = new LinkedList<String>(Arrays.asList(dataArray));
        if (dataList.get(0).equals("None")) {
            return null;
        }
        List<TreeNode> listNode = new ArrayList<TreeNode>();
        // 创建根节点
        TreeNode root = new TreeNode(Integer.valueOf(dataList.get(0)));
        dataList.remove(0);
        // 先把根节点放入链表中
        listNode.add(root);
        while (!dataList.isEmpty()) {
            // 从listNode的第一个元素开始往后遍历
            int listSize = listNode.size();
            for (int i = 0; i < listSize; i++) {
                TreeNode curNode = listNode.get(i);
                if (!dataList.get(0).equals("None")) {
                    curNode.left = new TreeNode(Integer.valueOf(dataList.get(0)));
                    dataList.remove(0);
                    listNode.add(curNode.left);
                } else {
                    curNode.left = null;
                    dataList.remove(0);
                }
                if (!dataList.get(0).equals("None")) {
                    curNode.right = new TreeNode(Integer.valueOf(dataList.get(0)));
                    dataList.remove(0);
                    listNode.add(curNode.right);
                } else {
                    curNode.right = null;
                    dataList.remove(0);
                }
            }
        }
        return root;
    }

    public TreeNode deserialize4(String data) {
        /*
         * 对deserialize3的一个优化
         */
        String[] dataArray = data.split(",");
        if (dataArray[0].equals("None")) {
            return null;
        }
        Deque<TreeNode> deque = new LinkedList<TreeNode>();
        // 创建根节点
        TreeNode root = new TreeNode(Integer.valueOf(dataArray[0]));
        deque.offer(root);
        int i = 1;
        while (!deque.isEmpty()) {
            TreeNode curNode = deque.poll();
            if (!dataArray[i].equals("None")) {
                curNode.left = new TreeNode(Integer.valueOf(dataArray[i]));
                deque.offer(curNode.left);
            }
            i++;
            if (!dataArray[i].equals("None")) {
                curNode.right = new TreeNode(Integer.valueOf(dataArray[i]));
                deque.offer(curNode.right);
            }
            i++;
        }
        return root;
    }
}
```

- Deque.offer()：将指定元素插入此双端队列表示的队列中（换句话说，在此双端队列的尾部）。
- Deque.poll()：检索并删除此双端队列表示的队列的头部（换句话说，此双端队列的第一个元素)，如果此双端队列为空，则返回null
