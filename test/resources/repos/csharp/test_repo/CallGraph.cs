namespace TestProject
{
    public static class CallGraph
    {
        public static void Leaf()
        {
        }

        public static void Mid()
        {
            Leaf();
        }

        public static void EntryA()
        {
            Mid();
        }

        public static void EntryB()
        {
            Mid();
        }
    }
}
