extern void platform_sim_setup(void);
extern int  stub_entry(int argc, char *argv[]);
extern void stub_exit(void);

int main(int argc, char *argv[])
{
    platform_sim_setup();
    int rc = stub_entry(argc, argv);
    stub_exit();
    return rc;
}
